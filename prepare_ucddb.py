import os
from datetime import datetime, timedelta
from glob import glob

import biosppy.signals.tools as st
import numpy as np
import pandas as pd
import pyedflib
from biosppy.signals.ecg import correct_rpeaks, hamilton_segmenter
from scipy.interpolate import splev, splrep
from scipy.signal import medfilt


data_dir = r"F:\datasets\UCDDB" # change this to your own path
save_dir = r"dataset/ucddb"
os.makedirs(save_dir, exist_ok=True)

interp_fs = 3
interp_duration = 60  # 1 minute
hr_min, hr_max = 20, 300

resp_event_types = {
    "HYP-O", "HYP-C", "HYP-M",
    "APNEA-O", "APNEA-C", "APNEA-M",
}


def minmax_scale(arr):
    arr = np.asarray(arr, dtype=np.float32)
    mn, mx = np.min(arr), np.max(arr)
    if mx - mn == 0:
        return np.zeros_like(arr, dtype=np.float32)
    return (arr - mn) / (mx - mn)


def build_x_from_features(feature_list):
    """
    feature_list: list of [((rri_tm, rri_signal), (ampl_tm, ampl_signal)), ...]
    return: (N, 2, 180)
    """
    fixed_time = np.arange(0, interp_duration, step=1 / float(interp_fs))
    features = []

    for (rri_pack, ampl_pack) in feature_list:
        rri_tm, rri_signal = rri_pack
        ampl_tm, ampl_signal = ampl_pack

        if len(rri_tm) < 4 or len(ampl_tm) < 4:
            continue

        try:
            rri_interp = splev(fixed_time, splrep(rri_tm, minmax_scale(rri_signal), k=3), ext=1)
            ampl_interp = splev(fixed_time, splrep(ampl_tm, minmax_scale(ampl_signal), k=3), ext=1)
            features.append([rri_interp, ampl_interp])
        except Exception:
            continue

    if not features:
        return np.empty((0, 2, len(fixed_time)), dtype=np.float32)

    return np.asarray(features, dtype=np.float32)


def find_ecg_channel(signal_labels):
    labels_lower = [label.lower() for label in signal_labels]
    priority_keywords = ["ecg", "ekg", "ecg1", "ecg2", "lead", "ii", "i", "v1", "v2"]

    for index, label in enumerate(labels_lower):
        if "ecg" in label or "ekg" in label:
            return index

    for keyword in priority_keywords:
        for index, label in enumerate(labels_lower):
            if keyword == label or keyword in label:
                return index

    raise ValueError(f"ECG channel not found. Available channels: {signal_labels}")


def recheck_date(record_start_time, current_time):
    current_time = current_time.replace(
        year=record_start_time.year,
        month=record_start_time.month,
        day=record_start_time.day,
    )
    diff = current_time - record_start_time
    if diff.total_seconds() < -12 * 3600:
        current_time += timedelta(days=1)
    elif diff.total_seconds() > 12 * 3600:
        current_time -= timedelta(days=1)
    return current_time


def load_event_df(label_path):
    columns = [
        "Time", "Type", "Duration", "Low", "Desaturation",
        "Snore", "Arousal", "B/T", "Rate Change",
    ]
    with open(label_path, "r", encoding="utf-8", errors="ignore") as handle:
        lines = handle.readlines()

    data_lines = lines[3:]
    rows = [line.strip().split() for line in data_lines if line.strip()]
    if not rows:
        return pd.DataFrame(columns=columns)

    return pd.DataFrame(rows, columns=columns)


def build_minute_labels(record_start_time, total_seconds, event_df):
    sec_label = np.zeros(int(total_seconds), dtype=np.uint8)

    if len(event_df) > 0:
        event_df = event_df[event_df["Type"].isin(resp_event_types)].reset_index(drop=True)

        for index in range(len(event_df)):
            try:
                start_time = datetime.strptime(event_df.iloc[index]["Time"], "%H:%M:%S")
                start_time = recheck_date(record_start_time, start_time)
                duration = int(float(event_df.iloc[index]["Duration"]))
            except Exception:
                continue

            start_sec = int((start_time - record_start_time).total_seconds())
            end_sec = start_sec + duration
            start_sec = max(0, start_sec)
            end_sec = min(int(total_seconds), end_sec)

            if end_sec > start_sec:
                sec_label[start_sec:end_sec] = 1

    n_minutes = int(total_seconds // 60)
    minute_labels = np.zeros(n_minutes, dtype=np.int64)
    for minute_index in range(n_minutes):
        start = minute_index * 60
        end = start + 60
        if np.any(sec_label[start:end] > 0):
            minute_labels[minute_index] = 1

    return minute_labels


def extract_rri_rpa_from_minute(signal, fs):
    signal, _, _ = st.filter_signal(
        signal,
        ftype="FIR",
        band="bandpass",
        order=int(0.3 * fs),
        frequency=[5, 30],
        sampling_rate=fs,
    )

    rpeaks, = hamilton_segmenter(signal, sampling_rate=fs)
    rpeaks, = correct_rpeaks(signal, rpeaks=rpeaks, sampling_rate=fs, tol=0.1)

    if len(rpeaks) < 4 or len(rpeaks) < 40 or len(rpeaks) > 200:
        return None

    rri_tm = rpeaks[1:] / float(fs)
    rri_signal = np.diff(rpeaks) / float(fs)
    rri_signal = medfilt(rri_signal, kernel_size=3)

    ampl_tm = rpeaks / float(fs)
    ampl_signal = signal[rpeaks]

    hr = 60.0 / rri_signal
    if not np.all((hr >= hr_min) & (hr <= hr_max)):
        return None

    return (rri_tm, rri_signal), (ampl_tm, ampl_signal)


def process_subject(rec_path, event_path, save_dir):
    subject = os.path.basename(rec_path).split(".")[0]
    print(f"Processing {subject} ...")

    edf = pyedflib.EdfReader(rec_path)
    signal_labels = edf.getSignalLabels()
    sample_rates = edf.getSampleFrequencies()
    record_start_time = edf.getStartdatetime()
    duration_sec = edf.getFileDuration()

    ecg_idx = find_ecg_channel(signal_labels)
    ecg_label = signal_labels[ecg_idx]
    fs = float(sample_rates[ecg_idx])
    ecg = edf.readSignal(ecg_idx).astype(np.float32)
    edf.close()

    event_df = load_event_df(event_path)
    minute_labels = build_minute_labels(record_start_time, duration_sec, event_df)

    total_minutes_from_signal = int(len(ecg) // (fs * 60))
    n_minutes = min(len(minute_labels), total_minutes_from_signal)

    minute_labels = minute_labels[:n_minutes]
    ecg = ecg[: int(n_minutes * fs * 60)]

    feature_list = []
    labels = []
    groups = []
    samples_per_minute = int(fs * 60)

    for minute_index in range(n_minutes):
        start = int(minute_index * samples_per_minute)
        end = int((minute_index + 1) * samples_per_minute)
        minute_signal = ecg[start:end]

        if len(minute_signal) != samples_per_minute:
            continue

        feature = extract_rri_rpa_from_minute(minute_signal, fs)
        if feature is None:
            continue

        feature_list.append(feature)
        labels.append(int(minute_labels[minute_index]))
        groups.append(subject)

    x = build_x_from_features(feature_list)
    y = np.asarray(labels[: len(x)], dtype=np.int64)
    groups = np.asarray(groups[: len(x)])

    save_path = os.path.join(save_dir, f"{subject}.npz")
    np.savez_compressed(save_path, x=x, y=y, groups=groups)
    print(f"{subject}: ECG channel={ecg_label}, fs={fs}, samples={len(x)}, saved to {save_path}")


def main():
    rec_files = sorted(glob(os.path.join(data_dir, "*.rec")))
    event_files = sorted(glob(os.path.join(data_dir, "*_respevt.txt")))

    event_map = {os.path.basename(path).split("_")[0]: path for path in event_files}

    for rec_path in rec_files:
        subject = os.path.basename(rec_path).split(".")[0]
        if subject not in event_map:
            print(f"[Skip] {subject}: missing matching _respevt.txt")
            continue

        event_path = event_map[subject]
        try:
            process_subject(rec_path, event_path, save_dir)
        except Exception as exc:
            print(f"[Error] {subject}: {exc}")


if __name__ == "__main__":
    main()
