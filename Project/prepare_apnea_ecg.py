import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count

import biosppy.signals.tools as st
import numpy as np
import wfdb
from biosppy.signals.ecg import correct_rpeaks, hamilton_segmenter
from scipy.interpolate import splev, splrep
from scipy.signal import medfilt
from tqdm import tqdm

base_dir = r"F:\datasets\apnea-ecg-database-1.0.0"  # change this to your local path
target_dir = "dataset/Apnea_ECG"
train_dir = os.path.join(target_dir, "train")
test_dir = os.path.join(target_dir, "test")

fs = 100
sample = fs * 60  # 1 minute
interp_fs = 3
interp_duration = 60  # 1 minute
hr_min, hr_max = 20, 300
num_worker = 10 if cpu_count() > 10 else max(1, cpu_count() - 1)


def minmax_scale(arr):
    mn, mx = np.min(arr), np.max(arr)
    if mx - mn == 0:
        return np.zeros_like(arr)
    return (arr - mn) / (mx - mn)


def build_x(feature_list):
    """
    Build fixed-length 1-minute inputs from a subject's extracted features.

    Returns:
        np.ndarray with shape (N, 2, 180)
    """
    fixed_time = np.arange(0, interp_duration, step=1 / float(interp_fs))
    features = []

    for (rri_tm, rri_signal), (ampl_tm, ampl_signal) in feature_list:
        if len(rri_tm) < 4 or len(ampl_tm) < 4:
            continue

        try:
            rri_interp = splev(fixed_time, splrep(rri_tm, minmax_scale(rri_signal), k=3), ext=1)
            ampl_interp = splev(fixed_time, splrep(ampl_tm, minmax_scale(ampl_signal), k=3), ext=1)
        except Exception:
            continue

        features.append([rri_interp, ampl_interp])

    if not features:
        return np.empty((0, 2, len(fixed_time)), dtype=np.float32)

    return np.asarray(features, dtype=np.float32)


def worker(name, labels):
    """
    Extract minute-level RRI/RPA features for one subject.

    Returns:
        subject_name, raw_features, labels
    """
    try:
        raw_features = []
        targets = []

        signal = wfdb.rdrecord(os.path.join(base_dir, name), channels=[0]).p_signal[:, 0]
        total_minutes = min(len(labels), int(len(signal) / float(sample)))

        for minute_idx in tqdm(range(total_minutes), desc=name, file=sys.stdout):
            start = minute_idx * sample
            end = (minute_idx + 1) * sample
            minute_signal = signal[start:end]

            if len(minute_signal) != sample:
                continue

            minute_signal, _, _ = st.filter_signal(
                minute_signal,
                ftype="FIR",
                band="bandpass",
                order=int(0.3 * fs),
                frequency=[5, 30],
                sampling_rate=fs,
            )

            rpeaks, = hamilton_segmenter(minute_signal, sampling_rate=fs)
            rpeaks, = correct_rpeaks(minute_signal, rpeaks=rpeaks, sampling_rate=fs, tol=0.1)

            if len(rpeaks) < 4 or len(rpeaks) < 40 or len(rpeaks) > 200:
                continue

            rri_tm = rpeaks[1:] / float(fs)
            rri_signal = np.diff(rpeaks) / float(fs)
            rri_signal = medfilt(rri_signal, kernel_size=3)

            ampl_tm = rpeaks / float(fs)
            ampl_signal = minute_signal[rpeaks]

            hr = 60.0 / rri_signal
            if not np.all(np.logical_and(hr >= hr_min, hr <= hr_max)):
                continue

            raw_features.append([(rri_tm, rri_signal), (ampl_tm, ampl_signal)])
            targets.append(0 if labels[minute_idx] == "N" else 1)

        return name, raw_features, targets

    except Exception as exc:
        print(f"Error processing file {name}: {exc}")
        raise


def save_subject(subject_name, raw_features, targets, save_dir):
    """
    Save one subject to a compressed npz file.
    """
    x = build_x(raw_features)
    y = np.asarray(targets[: len(x)], dtype=np.int64)

    if x.shape[0] != len(y):
        raise ValueError(f"Feature/label length mismatch for {subject_name}: {x.shape[0]} vs {len(y)}")

    save_path = os.path.join(save_dir, f"{subject_name}.npz")
    np.savez_compressed(save_path, x=x, y=y)
    print(f"Saved {subject_name}: x={x.shape}, y={y.shape} -> {save_path}")


if __name__ == "__main__":
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    train_names = [
        "a01", "a02", "a03", "a04", "a05", "a06", "a07", "a08", "a09", "a10",
        "a11", "a12", "a13", "a14", "a15", "a16", "a17", "a18", "a19", "a20",
        "b01", "b02", "b03", "b04", "b05",
        "c01", "c02", "c03", "c04", "c05", "c06", "c07", "c08", "c09", "c10",
    ]

    print("Train subjects generating...")
    with ProcessPoolExecutor(max_workers=num_worker) as executor:
        task_list = []
        for name in train_names:
            labels = wfdb.rdann(os.path.join(base_dir, name), extension="apn").symbol
            task_list.append(executor.submit(worker, name, labels))

        for task in as_completed(task_list):
            name, raw_features, targets = task.result()
            save_subject(name, raw_features, targets, train_dir)

    print("Train subjects done!")

    print("Loading test answers...")
    answers = {}
    with open(os.path.join("dataset", "event-2-answers"), "r") as handle:
        for answer in handle.read().strip().split("\n\n"):
            answers[answer[:3]] = list("".join(answer.split()[2::2]))

    test_names = [
        "x01", "x02", "x03", "x04", "x05", "x06", "x07", "x08", "x09", "x10",
        "x11", "x12", "x13", "x14", "x15", "x16", "x17", "x18", "x19", "x20",
        "x21", "x22", "x23", "x24", "x25", "x26", "x27", "x28", "x29", "x30",
        "x31", "x32", "x33", "x34", "x35",
    ]

    print("Test subjects generating...")
    with ProcessPoolExecutor(max_workers=num_worker) as executor:
        task_list = []
        for name in test_names:
            labels = answers[name]
            task_list.append(executor.submit(worker, name, labels))

        for task in as_completed(task_list):
            name, raw_features, targets = task.result()
            save_subject(name, raw_features, targets, test_dir)

    print("Test subjects done!")
