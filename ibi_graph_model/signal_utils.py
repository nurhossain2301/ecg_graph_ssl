import os
import numpy as np
import pandas as pd
import torch


def load_ibi_file(file_path):
    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".npy":
        x = np.load(file_path)

    elif ext == ".npz":
        data = np.load(file_path)
        # use first array in archive
        key = list(data.keys())[0]
        x = data[key]

    elif ext == ".pt":
        x = torch.load(file_path)
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
        else:
            x = np.asarray(x)

    elif ext in [".csv", ".txt"]:
        df = pd.read_csv(file_path)
        if df.shape[1] == 1:
            x = df.iloc[:, 0].values
        else:
            # if there is an obvious ibi column, use it
            lower_cols = [c.lower() for c in df.columns]
            if "ibi" in lower_cols:
                x = df.iloc[:, lower_cols.index("ibi")].values
            elif "rr" in lower_cols:
                x = df.iloc[:, lower_cols.index("rr")].values
            else:
                x = df.iloc[:, 0].values
    else:
        raise ValueError(f"Unsupported IBI file type: {file_path}")

    x = np.asarray(x).astype(np.float32).reshape(-1)

    # remove nan/inf
    x = x[np.isfinite(x)]

    return x


def maybe_convert_ms_to_sec(ibi):
    """
    Heuristic:
    if median IBI > 10, assume milliseconds and convert to seconds.
    """
    if len(ibi) == 0:
        return ibi
    med = np.median(ibi)
    if med > 10.0:
        return ibi / 1000.0
    return ibi


def clean_ibi(ibi, min_ibi_ms=300, max_ibi_ms=2000):
    """
    Robust but simple cleanup.
    Keeps sequence length unchanged as much as possible.
    """
    ibi = maybe_convert_ms_to_sec(ibi)
    ibi_ms = ibi * 1000.0

    # validity by physiology
    valid = (ibi_ms >= min_ibi_ms) & (ibi_ms <= max_ibi_ms)

    # fill bad values by local median
    x = ibi.copy()
    if len(x) == 0:
        return np.zeros(1, dtype=np.float32), np.zeros(1, dtype=np.float32)

    for i in range(len(x)):
        if not valid[i]:
            left = max(0, i - 2)
            right = min(len(x), i + 3)
            neigh = x[left:right]
            neigh = neigh[np.isfinite(neigh)]
            if len(neigh) > 0:
                x[i] = np.median(neigh)
            else:
                x[i] = np.median(x[np.isfinite(x)]) if np.isfinite(x).any() else 1.0

    # recompute reliability
    ibi_ms = x * 1000.0
    quality = ((ibi_ms >= min_ibi_ms) & (ibi_ms <= max_ibi_ms)).astype(np.float32)

    return x.astype(np.float32), quality.astype(np.float32)


_N_FEATURES = 10


def _mad_norm(x):
    med = np.median(x)
    mad = np.median(np.abs(x - med)) + 1e-6
    return (x - med) / mad


def build_ibi_features(ibi_sec, quality, max_len_beats=128):
    """
    Node features per beat interval (10 features):
      0: ibi_norm       — MAD-normalised IBI
      1: dibi_norm      — MAD-normalised delta-IBI (successive difference)
      2: local_var_norm — MAD-normalised local std over ±2-beat window
      3: quality        — signal quality flag [0, 1]
      4: hr_norm        — MAD-normalised instantaneous heart rate (60 / IBI)
      5: abs_dibi_norm  — MAD-normalised |delta-IBI|  (per-beat RMSSD proxy)
      6: pnn50_flag     — 1 if |delta-IBI| > 50 ms, else 0
      7: rel_ibi        — IBI / median_IBI  (ratio; ~1 for normal beats)
      8: trend_norm     — MAD-normalised local linear slope over ±2-beat window
                          (positive = lengthening, negative = shortening)
      9: position       — normalised beat position in window  [0, 1]
    """
    if len(ibi_sec) == 0:
        return np.zeros((1, _N_FEATURES), dtype=np.float32)

    ibi     = ibi_sec.astype(np.float32)[:max_len_beats]
    quality = quality[:max_len_beats]
    N       = len(ibi)

    # 0: ibi_norm
    ibi_norm = _mad_norm(ibi)

    # 1: dibi_norm
    dibi      = np.diff(ibi, prepend=ibi[0])
    dibi_norm = _mad_norm(dibi)

    # 2: local_var_norm
    local_var = np.array([
        np.std(ibi[max(0, i - 2):min(N, i + 3)])
        for i in range(N)
    ], dtype=np.float32)
    local_var_norm = _mad_norm(local_var)

    # 3: quality  (pass-through)

    # 4: hr_norm  (clip IBI to 0.2–3 s → 20–300 BPM before dividing)
    hr      = 60.0 / np.clip(ibi, 0.2, 3.0)
    hr_norm = _mad_norm(hr)

    # 5: abs_dibi_norm  (|ΔlBI|, RMSSD proxy)
    abs_dibi      = np.abs(dibi)
    abs_dibi_norm = _mad_norm(abs_dibi)

    # 6: pnn50_flag  (|ΔIBI| > 50 ms)
    pnn50_flag = (abs_dibi > 0.05).astype(np.float32)

    # 7: rel_ibi  (ratio to median; scale-free)
    rel_ibi = ibi / (np.median(ibi) + 1e-6)

    # 8: trend_norm  (local OLS slope over ±2-beat window)
    trend = np.zeros(N, dtype=np.float32)
    for i in range(N):
        l, r = max(0, i - 2), min(N, i + 3)
        seg  = ibi[l:r]
        if len(seg) > 1:
            xs = np.arange(len(seg), dtype=np.float32) - (len(seg) - 1) / 2.0
            trend[i] = np.dot(xs, seg - seg.mean()) / (np.dot(xs, xs) + 1e-6)
    trend_norm = _mad_norm(trend)

    # 9: position  (normalised index in window)
    position = np.linspace(0.0, 1.0, N, dtype=np.float32)

    feats = np.stack(
        [ibi_norm, dibi_norm, local_var_norm, quality,
         hr_norm, abs_dibi_norm, pnn50_flag, rel_ibi,
         trend_norm, position],
        axis=-1,
    ).astype(np.float32)

    return feats


_HRV_FEATURE_DIM = 11
_HRV_FEATURE_NAMES = [
    "meanNN", "SDNN", "RMSSD", "pNN50",
    "HF_power", "LF_power", "LF_HF_ratio",
    "RSA", "HR_range", "SD1", "S"
]


def compute_hrv_features(ibi_sec, fs_interp=5.0):
    """
    Window-level HRV feature vector (11 features) from cleaned IBI in seconds.

    Features (index → name):
      0  meanNN      mean RR interval (ms)
      1  SDNN        std of RR intervals (ms)
      2  RMSSD       sqrt(mean successive diff²) (ms)
      3  pNN50       % successive diffs > 50 ms
      4  HF_power    Welch PSD integral 0.24–1.04 Hz (infant HF band)
      5  LF_power    Welch PSD integral 0.04–0.15 Hz
      6  LF_HF_ratio LF / HF
      7  RSA         log variance of band-passed HP in HF band
      8  HR_range    max(HR) − min(HR) (bpm)
      9  SD1         Poincaré short-term variability (ms)
     10  S           Poincaré area π·SD1·SD2 (ms²)
    """
    from scipy import signal as sp_signal
    from scipy.interpolate import interp1d

    N = len(ibi_sec)
    ibi_ms = ibi_sec * 1000.0

    # ---- time domain ----
    meanNN = float(np.mean(ibi_ms))
    SDNN = float(np.std(ibi_ms))
    diff_ms = np.diff(ibi_ms) if N > 1 else np.zeros(1, dtype=np.float32)
    RMSSD = float(np.sqrt(np.mean(diff_ms ** 2)))
    pNN50 = float(np.mean(np.abs(diff_ms) > 50.0) * 100.0)

    # ---- HR range ----
    hr = 60.0 / np.clip(ibi_sec, 0.1, 3.0)
    HR_range = float(np.max(hr) - np.min(hr))

    # ---- Poincaré ----
    if N > 1:
        sd1 = float(np.std(diff_ms) / np.sqrt(2.0))
        sd2 = float(np.sqrt(max(2.0 * SDNN ** 2 - sd1 ** 2, 0.0)))
        S = float(np.pi * sd1 * sd2)
    else:
        sd1 = sd2 = S = 0.0

    # ---- frequency domain ----
    HF_power = LF_power = RSA = 0.0

    if N >= 8:
        t_beats = np.concatenate([[0.0], np.cumsum(ibi_sec[:-1].astype(np.float64))])
        duration = t_beats[-1] + float(ibi_sec[-1])
        t_reg = np.arange(0.0, duration, 1.0 / fs_interp)

        if len(t_reg) >= 16:
            f_interp = interp1d(
                t_beats, ibi_sec.astype(np.float64), kind="linear",
                bounds_error=False,
                fill_value=(float(ibi_sec[0]), float(ibi_sec[-1]))
            )
            hp = f_interp(t_reg).astype(np.float64)
            hp -= hp.mean()

            # need fs/0.04 = 125 pts for 0.04 Hz frequency resolution
            nperseg = min(len(hp), max(16, int(fs_interp / 0.04)))
            freqs, psd = sp_signal.welch(hp, fs=fs_interp, nperseg=nperseg)
            df = float(freqs[1] - freqs[0]) if len(freqs) > 1 else 1.0

            hf_mask = (freqs >= 0.24) & (freqs <= 1.04)
            lf_mask = (freqs >= 0.04) & (freqs <= 0.15)
            HF_power = float(np.sum(psd[hf_mask]) * df) if hf_mask.any() else 0.0
            LF_power = float(np.sum(psd[lf_mask]) * df) if lf_mask.any() else 0.0

            try:
                nyq = fs_interp / 2.0
                low_n, high_n = 0.24 / nyq, min(1.04 / nyq, 0.99)
                if low_n < high_n and len(hp) > 12:
                    b, a = sp_signal.butter(2, [low_n, high_n], btype="band")
                    hp_bp = sp_signal.filtfilt(b, a, hp)
                    RSA = float(np.log(np.var(hp_bp) + 1e-10))
                else:
                    RSA = float(np.log(max(HF_power, 1e-10)))
            except Exception:
                RSA = float(np.log(max(HF_power, 1e-10)))

    LF_HF_ratio = LF_power / (HF_power + 1e-10)

    return np.array(
        [meanNN, SDNN, RMSSD, pNN50, HF_power, LF_power, LF_HF_ratio, RSA, HR_range, sd1, S],
        dtype=np.float32,
    )


def random_window_ibi(ibi, window_beats=128, train=True, val_window_idx=0, val_windows_total=5):
    if len(ibi) <= window_beats:
        return ibi

    if train:
        start = np.random.randint(0, len(ibi) - window_beats + 1)
    else:
        # evenly spaced deterministic windows across the sequence
        max_start = len(ibi) - window_beats
        start = int((val_window_idx + 1) * max_start / (val_windows_total + 1))
        start = max(0, min(start, max_start))

    return ibi[start:start + window_beats]