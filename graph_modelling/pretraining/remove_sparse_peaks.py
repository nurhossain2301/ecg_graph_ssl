"""
Remove files with fewer than --min_peaks peaks from the train/val CSVs
and delete their _peaks.npy files.

Reads validate_peaks_report.csv (produced by validate_peaks.py).
Run validate_peaks.py first if the report is stale.

Usage
-----
  # dry-run (shows what would be removed, touches nothing)
  python remove_sparse_peaks.py --dry_run

  # actually remove
  python remove_sparse_peaks.py
"""

import os
import argparse
import pandas as pd

REPO_ROOT     = "/work/nvme/bebr/mkhan14/ecg_foundation_model"
DEFAULT_REPORT = os.path.join(REPO_ROOT, "graph_modelling/pretraining/validate_peaks_report.csv")
DEFAULT_TRAIN  = "/work/hdd/bebr/Projects/ecg_foundational_model/ECG_train_files.csv"
DEFAULT_VAL    = "/work/hdd/bebr/Projects/ecg_foundational_model/ECG_val_files.csv"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--report",    default=DEFAULT_REPORT)
    parser.add_argument("--train_csv", default=DEFAULT_TRAIN)
    parser.add_argument("--val_csv",   default=DEFAULT_VAL)
    parser.add_argument("--min_peaks", type=int, default=5,
                        help="Remove files with fewer than this many peaks")
    parser.add_argument("--dry_run",   action="store_true",
                        help="Print what would be removed without changing anything")
    args = parser.parse_args()

    report = pd.read_csv(args.report)

    # files that have a valid peaks file but too few peaks
    to_remove = report[
        (report["status"].isin(["ok", "warning"])) &
        (report["n_peaks"] < args.min_peaks)
    ]["wav"].tolist()

    print(f"Min peaks threshold : {args.min_peaks}")
    print(f"Files to remove     : {len(to_remove)}")

    if len(to_remove) == 0:
        print("Nothing to remove.")
        return

    for wav in to_remove:
        peaks = wav.replace(".wav", "_peaks.npy")
        print(f"  WAV   : {wav}")
        print(f"  peaks : {peaks}  (exists={os.path.exists(peaks)})")

    if args.dry_run:
        print("\n[dry-run] No files changed.")
        return

    # ── remove from CSVs ──────────────────────────────────────────────────────
    remove_set = set(to_remove)
    for csv_path in [args.train_csv, args.val_csv]:
        if not os.path.isfile(csv_path):
            continue
        df = pd.read_csv(csv_path)
        before = len(df)
        df = df[~df["filename"].isin(remove_set)]
        after = len(df)
        df.to_csv(csv_path, index=False)
        print(f"  {csv_path}: {before} → {after} rows (removed {before - after})")

    # ── delete _peaks.npy files ───────────────────────────────────────────────
    for wav in to_remove:
        peaks = wav.replace(".wav", "_peaks.npy")
        if os.path.exists(peaks):
            os.remove(peaks)
            print(f"  Deleted: {peaks}")

    print(f"\nDone. Removed {len(to_remove)} file(s).")


if __name__ == "__main__":
    main()
