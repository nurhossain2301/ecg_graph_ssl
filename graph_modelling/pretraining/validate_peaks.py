"""
Validate all pre-computed _peaks.npy files.

Checks per file
---------------
CRITICAL (marks file as bad):
  - peaks file missing
  - numpy load error (corrupted)
  - array is not 1-D int64
  - fewer than 2 peaks (can't compute IBI)
  - peaks not strictly sorted
  - any peak < 0 or >= n_samples

WARNING (file usable but suspicious):
  - fewer than 5 peaks (very short/sparse signal)
  - >20 % of IBIs outside physiological range [200, 1500] ms
  - median IBI outside normal infant range [300, 800] ms
  - detected HR outside [40, 200] BPM

Usage
-----
  python validate_peaks.py [--num_workers N] [--report PATH]
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import soundfile as sf
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

REPO_ROOT = "/work/nvme/bebr/mkhan14/ecg_foundation_model"
DEFAULT_TRAIN_CSV = "/work/hdd/bebr/Projects/ecg_foundational_model/ECG_train_files.csv"
DEFAULT_VAL_CSV   = "/work/hdd/bebr/Projects/ecg_foundational_model/ECG_val_files.csv"
DEFAULT_REPORT    = os.path.join(REPO_ROOT, "graph_modelling/pretraining/validate_peaks_report.csv")

# Physiological thresholds (infant ECG, sr=1000)
IBI_MIN_MS       = 200    # 300 BPM absolute ceiling
IBI_MAX_MS       = 1500   # 40 BPM absolute floor
IBI_NORMAL_MIN   = 300    # normal infant low end
IBI_NORMAL_MAX   = 800    # normal infant high end
BAD_IBI_FRACTION = 0.20   # flag if >20% of IBIs outside [MIN, MAX]


def validate_file(wav_path: str) -> dict:
    peaks_path = wav_path.replace(".wav", "_peaks.npy")

    row = {
        "wav":          wav_path,
        "peaks_path":   peaks_path,
        "status":       "ok",          # ok | warning | critical
        "issues":       "",
        "n_peaks":      -1,
        "dur_sec":      -1.0,
        "sr":           -1,
        "n_samples":    -1,
        "median_ibi_ms":  float("nan"),
        "min_ibi_ms":     float("nan"),
        "max_ibi_ms":     float("nan"),
        "bad_ibi_frac":   float("nan"),
        "mean_hr_bpm":    float("nan"),
    }

    issues   = []
    critical = False

    # ── 1. peaks file exists ──────────────────────────────────────────────────
    if not os.path.exists(peaks_path):
        row["status"] = "critical"
        row["issues"] = "MISSING peaks file"
        return row

    # ── 2. load peaks ─────────────────────────────────────────────────────────
    try:
        peaks = np.load(peaks_path)
    except Exception as e:
        row["status"] = "critical"
        row["issues"] = f"LOAD ERROR: {e}"
        return row

    # ── 3. array sanity ───────────────────────────────────────────────────────
    if peaks.ndim != 1:
        issues.append(f"BAD SHAPE {peaks.shape} (expected 1-D)"); critical = True
    if peaks.dtype != np.int64:
        issues.append(f"BAD DTYPE {peaks.dtype} (expected int64)"); critical = True

    if critical:
        row["status"] = "critical"
        row["issues"] = " | ".join(issues)
        return row

    row["n_peaks"] = int(len(peaks))

    # ── 4. minimum peaks ──────────────────────────────────────────────────────
    if len(peaks) < 2:
        row["status"] = "critical"
        row["issues"] = f"TOO FEW PEAKS: {len(peaks)}"
        return row

    # ── 5. get WAV metadata (fast, no full decode) ────────────────────────────
    try:
        info = sf.info(wav_path)
        sr         = info.samplerate
        n_samples  = info.frames
        dur_sec    = info.duration
    except Exception as e:
        issues.append(f"WAV INFO ERROR: {e}"); critical = True
        row["status"] = "critical"
        row["issues"] = " | ".join(issues)
        return row

    row["sr"]        = int(sr)
    row["n_samples"] = int(n_samples)
    row["dur_sec"]   = round(float(dur_sec), 2)

    # ── 6. peaks in valid range ───────────────────────────────────────────────
    if peaks[0] < 0:
        issues.append(f"NEGATIVE PEAK: {peaks[0]}"); critical = True
    if peaks[-1] >= n_samples:
        issues.append(f"PEAK >= n_samples ({peaks[-1]} >= {n_samples})"); critical = True

    # ── 7. strictly sorted ────────────────────────────────────────────────────
    if not np.all(np.diff(peaks) > 0):
        n_bad = int((np.diff(peaks) <= 0).sum())
        issues.append(f"NOT SORTED: {n_bad} non-increasing steps"); critical = True

    if critical:
        row["status"] = "critical"
        row["issues"] = " | ".join(issues)
        return row

    # ── 8. IBI statistics ─────────────────────────────────────────────────────
    ibi_ms = np.diff(peaks).astype(np.float64) / sr * 1000.0

    median_ibi = float(np.median(ibi_ms))
    min_ibi    = float(ibi_ms.min())
    max_ibi    = float(ibi_ms.max())
    bad_frac   = float(((ibi_ms < IBI_MIN_MS) | (ibi_ms > IBI_MAX_MS)).mean())
    mean_hr    = 60_000.0 / median_ibi if median_ibi > 0 else float("nan")

    row["median_ibi_ms"] = round(median_ibi, 1)
    row["min_ibi_ms"]    = round(min_ibi, 1)
    row["max_ibi_ms"]    = round(max_ibi, 1)
    row["bad_ibi_frac"]  = round(bad_frac, 4)
    row["mean_hr_bpm"]   = round(mean_hr, 1)

    # ── 9. physiological warnings ─────────────────────────────────────────────
    if len(peaks) < 5:
        issues.append(f"SPARSE: only {len(peaks)} peaks")

    if bad_frac > BAD_IBI_FRACTION:
        issues.append(
            f"HIGH BAD-IBI FRACTION: {bad_frac:.1%} outside [{IBI_MIN_MS},{IBI_MAX_MS}]ms"
        )

    if not (IBI_NORMAL_MIN <= median_ibi <= IBI_NORMAL_MAX):
        issues.append(f"UNUSUAL MEDIAN IBI: {median_ibi:.0f}ms (normal {IBI_NORMAL_MIN}–{IBI_NORMAL_MAX}ms)")

    if not (40 <= mean_hr <= 200):
        issues.append(f"UNUSUAL HR: {mean_hr:.0f} BPM")

    if issues:
        row["status"] = "warning"
        row["issues"] = " | ".join(issues)
    else:
        row["status"] = "ok"

    return row


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
    parser.add_argument("--num_workers", type=int, default=min(32, cpu_count()))
    args = parser.parse_args()

    all_files = collect_files(args.train_csv, args.val_csv)
    print(f"Files to validate : {len(all_files)}")
    print(f"Workers           : {args.num_workers}\n")

    with Pool(args.num_workers) as pool:
        results = list(tqdm(pool.imap(validate_file, all_files), total=len(all_files)))

    df = pd.DataFrame(results)

    n_ok       = (df["status"] == "ok").sum()
    n_warn     = (df["status"] == "warning").sum()
    n_critical = (df["status"] == "critical").sum()

    print(f"\n{'='*50}")
    print(f"  SUMMARY")
    print(f"{'='*50}")
    print(f"  OK       : {n_ok}")
    print(f"  Warning  : {n_warn}")
    print(f"  Critical : {n_critical}")
    print(f"{'='*50}")

    if n_ok > 0:
        ok = df[df["status"] == "ok"]
        print(f"\n  Healthy-file stats:")
        print(f"    Median IBI  : {ok['median_ibi_ms'].median():.1f} ms")
        print(f"    Mean HR     : {ok['mean_hr_bpm'].mean():.1f} BPM")
        print(f"    Peaks/file  : {ok['n_peaks'].mean():.1f} (median {ok['n_peaks'].median():.0f})")
        print(f"    Duration    : {ok['dur_sec'].mean():.1f}s avg")

    if n_warn > 0:
        print(f"\n  WARNING files ({n_warn}):")
        for _, r in df[df["status"] == "warning"].iterrows():
            print(f"    {r['wav']}")
            print(f"      → {r['issues']}")

    if n_critical > 0:
        print(f"\n  CRITICAL files ({n_critical}):")
        for _, r in df[df["status"] == "critical"].iterrows():
            print(f"    {r['wav']}")
            print(f"      → {r['issues']}")

    df.to_csv(args.report, index=False)
    print(f"\n  Report → {args.report}")


if __name__ == "__main__":
    main()
