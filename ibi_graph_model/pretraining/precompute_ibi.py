#!/usr/bin/env python3
"""
Precompute IBI sequences from ECG files and save as .npz arrays (ibi + time).
The time array contains R-peak timestamps in seconds, enabling fixed-duration
time-based windowing in the dataloader instead of beat-count windowing.

Generates an output CSV with a "filename" column ready for the IBI dataloader.

Usage:
  python precompute_ibi.py \
      --input_csv   /path/to/ecg_files.csv \
      --ecg_col     filename \
      --output_dir  /path/to/ibi_npz/ \
      --output_csv  /path/to/ibi_files.csv \
      --num_workers 20 \
      --skip_done
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from littlebeats.littlebeats.ecg.ibi import detect_ibi_adaptive

SAMPLING_RATE = 1000


def _process_one(args):
    ecg_wav, output_dir = args
    try:
        ecg_txt = ecg_wav.replace(".wav", ".txt")
        ecg_df  = pd.read_csv(ecg_txt, names=["time", "ecg"])
        ecg_df.attrs["sr"] = SAMPLING_RATE

        ibi_df = detect_ibi_adaptive(
            ecg_df, "ecg", ["time"], debug=False, verbose=False
        )

        # ibi_df has columns: "time" (R-peak timestamp in seconds) and "ibi"
        ibi  = ibi_df["ibi"].values.astype(np.float32)
        time = ibi_df["time"].values.astype(np.float32)

        stem     = os.path.splitext(os.path.basename(ecg_wav))[0]
        out_path = os.path.join(output_dir, stem + ".npz")
        np.savez(out_path, ibi=ibi, time=time)

        n_beats    = len(ibi)
        duration_s = float(time[-1] - time[0]) if n_beats > 1 else 0.0
        wrong_mask = (ibi < 0.300) | (ibi > 2.000)
        wrong_pct  = float(wrong_mask.sum()) / max(n_beats, 1) * 100

        return {
            "ecg_file":   ecg_wav,
            "ibi_file":   out_path,
            "num_beats":  n_beats,
            "duration_s": round(duration_s, 1),
            "wrong_pct":  round(wrong_pct, 2),
            "status":     "ok",
        }
    except Exception as exc:
        return {
            "ecg_file":   ecg_wav,
            "ibi_file":   "",
            "num_beats":  0,
            "duration_s": 0.0,
            "wrong_pct":  100.0,
            "status":     f"error: {exc}",
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv",   required=True)
    parser.add_argument("--ecg_col",     default="filename",
                        help="Column name for ECG .wav paths")
    parser.add_argument("--output_dir",  required=True,
                        help="Directory to save .npz files (ibi + time)")
    parser.add_argument("--output_csv",  required=True,
                        help="Output CSV with filename → .npz path")
    parser.add_argument("--num_workers", type=int,
                        default=max(1, cpu_count() - 2))
    parser.add_argument("--skip_done",   action="store_true",
                        help="Skip files whose .npz already exists")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    df        = pd.read_csv(args.input_csv)
    all_files = df[args.ecg_col].tolist()

    if args.skip_done:
        todo = [
            f for f in all_files
            if not os.path.exists(
                os.path.join(args.output_dir,
                             os.path.splitext(os.path.basename(f))[0] + ".npz")
            )
        ]
        print(f"Already computed: {len(all_files) - len(todo)}  (will skip)")
    else:
        todo = all_files

    print(f"To compute      : {len(todo)}")
    if not todo:
        print("Nothing to do.")
        return

    results = []
    with Pool(args.num_workers) as pool:
        for res in tqdm(
            pool.imap_unordered(_process_one, [(f, args.output_dir) for f in todo]),
            total=len(todo),
        ):
            results.append(res)

    report = pd.DataFrame(results)
    ok     = report[report["status"] == "ok"]

    report_path = os.path.join(args.output_dir, "ibi_quality_report.csv")
    report.to_csv(report_path, index=False)

    out_df = ok[["ibi_file"]].rename(columns={"ibi_file": "filename"})
    out_df.to_csv(args.output_csv, index=False)

    print("\n===== SUMMARY =====")
    print(f"OK      : {len(ok)}")
    print(f"Errors  : {(report['status'] != 'ok').sum()}")
    if len(ok):
        print(f"Avg duration    : {ok['duration_s'].mean():.0f}s")
        print(f"Avg wrong IBI % : {ok['wrong_pct'].mean():.2f}")
        print(f"Files >20% wrong: {(ok['wrong_pct'] > 20).sum()} / {len(ok)}")
    print(f"\nReport  → {report_path}")
    print(f"IBI CSV → {args.output_csv}")


if __name__ == "__main__":
    main()
