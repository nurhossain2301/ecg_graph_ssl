import os
import numpy as np
import pandas as pd
import soundfile as sf
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import sys
sys.path.append("..")

from littlebeats.littlebeats.ecg.ibi import detect_ibi_simple


# -------------------------------------------------------
# CONFIG
# -------------------------------------------------------
INPUT_CSV = "/work/hdd/bebr/Projects/ecg_foundational_model/ECG_train_files.csv"        # CSV with column "filename"
PARENT_DIR = "ibi_outputs"
os.makedirs(PARENT_DIR, exist_ok=True)
OUTPUT_DIR = os.path.join(PARENT_DIR, "train")
NUM_WORKERS = max(1, cpu_count() - 2)

SAMPLING_RATE = 1000

# IBI thresholds (ms)
MIN_IBI = 350
MAX_IBI = 650

os.makedirs(OUTPUT_DIR, exist_ok=True)


# -------------------------------------------------------
# CORE FUNCTION
# -------------------------------------------------------
def process_file(file_path):

    # ---- Load ECG ----
    ecg_file = file_path.replace(".wav", ".txt")
    ecg_df = pd.read_csv(ecg_file, names=["time", "ecg"])
    ecg_df.attrs['sr'] = 1000
    

    ibi_df = detect_ibi_simple(
        ecg_df,
        'ecg',
        ['time'],
        debug=False,
        verbose=False)

    ibi = ibi_df['ibi'].values

    # ---- Detect wrong IBIs ----
    wrong_mask = (ibi < MIN_IBI) | (ibi > MAX_IBI)
    num_wrong = int(wrong_mask.sum())
    total = len(ibi)

    wrong_pct = (num_wrong / total) * 100 if total > 0 else 100.0

    # ---- Save IBI ----
    out_name = os.path.basename(file_path).replace(".wav", ".npy")
    out_path = os.path.join(OUTPUT_DIR, out_name)

    np.save(out_path, ibi.astype(np.float32))

    return {
        "file": file_path,
        "ibi_file": out_path,
        "num_beats": int(total),
        "num_wrong": num_wrong,
        "wrong_pct": round(wrong_pct, 2),
        "status": "ok"
    }

    


# -------------------------------------------------------
# MULTIPROCESS RUNNER
# -------------------------------------------------------
def run_parallel(file_list):

    results = []
    with Pool(NUM_WORKERS) as pool:
        for res in tqdm(pool.imap_unordered(process_file, file_list), total=len(file_list)):
            results.append(res)

    return results


# -------------------------------------------------------
# MAIN
# -------------------------------------------------------
def main():

    df = pd.read_csv(INPUT_CSV)
    file_list = df["filename"].tolist()

    print(f"Processing {len(file_list)} ECG files with {NUM_WORKERS} workers...")

    results = run_parallel(file_list)

    # ---- Save report ----
    report_df = pd.DataFrame(results)

    report_path = os.path.join(OUTPUT_DIR, "ibi_quality_report.csv")
    report_df.to_csv(report_path, index=False)

    # ---- Global stats ----
    valid = report_df[report_df["status"] == "ok"]

    if len(valid) > 0:
        print("\n===== SUMMARY =====")
        print(f"Avg wrong IBI %: {valid['wrong_pct'].mean():.2f}")
        print(f"Median wrong IBI %: {valid['wrong_pct'].median():.2f}")
        print(f"Files >20% wrong: {(valid['wrong_pct'] > 20).sum()} / {len(valid)}")

    print(f"\nSaved report → {report_path}")
    print(f"IBI files saved in → {OUTPUT_DIR}")


if __name__ == "__main__":
    main()