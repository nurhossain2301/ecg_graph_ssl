"""
Precompute R-peak sample indices for all ECG files used in pretraining.

Uses detect_ibi_adaptive (littlebeats) for high-quality multi-channel peak
detection. Output is saved as a sibling .npy file next to each .wav:
    /path/to/ecg.wav  →  /path/to/ecg_peaks.npy

The .npy file contains a 1-D int64 array of R-peak positions in samples.
The dataset can load this file and slice by window boundaries at train time.
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import soundfile as sf
from tqdm import tqdm
from multiprocessing import cpu_count

REPO_ROOT = "/work/nvme/bebr/mkhan14/ecg_foundation_model"
sys.path.insert(0, REPO_ROOT)
from littlebeats.littlebeats.ecg.ibi import detect_ibi_adaptive

DEFAULT_TRAIN_CSV = "/work/hdd/bebr/Projects/ecg_foundational_model/ECG_train_files.csv"
DEFAULT_VAL_CSV   = "/work/hdd/bebr/Projects/ecg_foundational_model/ECG_val_files.csv"
DEFAULT_REPORT    = os.path.join(REPO_ROOT, "graph_modelling/pretraining/precompute_peaks_report.csv")


def process_file(wav_path):
    peaks_path = wav_path.replace(".wav", "_peaks.npy")

    if os.path.exists(peaks_path):
        return {"file": wav_path, "status": "skipped", "n_peaks": -1, "error": ""}

    try:
        waveform, sr = sf.read(wav_path, dtype="float32")
        if waveform.ndim > 1:
            waveform = waveform[:, 0]

        n_samples = len(waveform)
        times = np.arange(n_samples, dtype=np.float64) / sr
        ecg_df = pd.DataFrame({"time": times, "ecg": waveform.astype(np.float64)})
        ecg_df.attrs["sr"] = sr

        ibi_df = detect_ibi_adaptive(ecg_df, "ecg", ["time"], debug=False, verbose=False)

        # ibi_df["time"] holds R-peak times in seconds → convert to sample indices
        peak_times   = ibi_df["time"].values
        peak_samples = np.round(peak_times * sr).astype(np.int64)
        peak_samples = peak_samples[(peak_samples >= 0) & (peak_samples < n_samples)]

        np.save(peaks_path, peak_samples)

        return {
            "file":     wav_path,
            "status":   "ok",
            "n_peaks":  len(peak_samples),
            "dur_sec":  round(n_samples / sr, 1),
            "avg_rr_ms": round(float(np.diff(peak_samples).mean()) / sr * 1000, 1) if len(peak_samples) > 1 else -1,
            "error":    "",
        }

    except Exception as exc:
        return {"file": wav_path, "status": "error", "n_peaks": 0, "dur_sec": -1, "avg_rr_ms": -1, "error": str(exc)}


def collect_files(train_csv, val_csv):
    dfs = []
    for path in [train_csv, val_csv]:
        if os.path.isfile(path):
            dfs.append(pd.read_csv(path)["filename"])
    files = pd.concat(dfs).dropna().unique().tolist()
    return [f for f in files if os.path.isfile(f)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_csv",   default=DEFAULT_TRAIN_CSV)
    parser.add_argument("--val_csv",     default=DEFAULT_VAL_CSV)
    parser.add_argument("--report",      default=DEFAULT_REPORT)
    parser.add_argument("--num_workers", type=int, default=max(1, cpu_count() - 2))
    args = parser.parse_args()

    all_files = collect_files(args.train_csv, args.val_csv)
    print(f"Files found     : {len(all_files)}")
    print(f"Workers         : {args.num_workers}")

    files_to_process = [f for f in all_files if not os.path.exists(f.replace(".wav", "_peaks.npy"))]
    already_done = len(all_files) - len(files_to_process)
    print(f"Already computed: {already_done}  (will skip)")
    print(f"To compute      : {len(files_to_process)}\n")

    results = [process_file(f) for f in tqdm(files_to_process)]

    skipped = [{"file": f, "status": "skipped", "n_peaks": -1, "error": ""} for f in all_files if os.path.exists(f.replace(".wav", "_peaks.npy"))]
    results.extend(skipped)

    report_df = pd.DataFrame(results)
    n_ok      = (report_df["status"] == "ok").sum()
    n_skipped = (report_df["status"] == "skipped").sum()
    n_errors  = (report_df["status"] == "error").sum()

    print(f"\n===== SUMMARY =====")
    print(f"OK      : {n_ok}")
    print(f"Skipped : {n_skipped}")
    print(f"Errors  : {n_errors}")

    if n_ok > 0:
        valid = report_df[report_df["status"] == "ok"]
        print(f"Avg peaks / file : {valid['n_peaks'].mean():.1f}")
        print(f"Avg RR (ms)      : {valid['avg_rr_ms'].mean():.1f}")

    if n_errors > 0:
        print("\nFailed files:")
        for _, row in report_df[report_df["status"] == "error"].iterrows():
            print(f"  {row['file']}\n    {row['error']}")

    report_df.to_csv(args.report, index=False)
    print(f"\nReport → {args.report}")


if __name__ == "__main__":
    main()
