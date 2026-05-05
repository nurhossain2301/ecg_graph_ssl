import os
import numpy as np
import pandas as pd
import torch
from scipy.signal import welch
from scipy.interpolate import interp1d


def load_ibi_with_time(file_path):
    data = np.load(file_path)
    ibi  = data["ibi"].astype(np.float32)
    time = data["time"].astype(np.float32)
    valid = np.isfinite(ibi) & np.isfinite(time)
    return ibi[valid], time[valid]


def window_ibi_by_time(ibi, time, window_sec=30, train=True,
                       val_window_idx=0, val_windows_total=100):
    if len(ibi) == 0:
        return ibi, time

    total_dur = float(time[-1] - time[0])
    if total_dur <= window_sec:
        return ibi, time

    max_start = total_dur - window_sec

    if train:
        t_start = float(time[0]) + np.random.uniform(0.0, max_start)
    else:
        t_start = float(time[0]) + (val_window_idx + 1) * max_start / (val_windows_total + 1)
        t_start = float(np.clip(t_start, time[0], float(time[0]) + max_start))

    t_end = t_start + window_sec
    mask  = (time >= t_start) & (time < t_end)
    return ibi[mask], time[mask]


def maybe_convert_ms_to_sec(ibi):
    if len(ibi) == 0:
        return ibi
    if np.median(ibi) > 10.0:
        return ibi / 1000.0
    return ibi


def clean_ibi(ibi, min_ibi_ms=300, max_ibi_ms=2000):
    ibi = maybe_convert_ms_to_sec(ibi)
    ibi_ms = ibi * 1000.0
    valid = (ibi_ms >= min_ibi_ms) & (ibi_ms <= max_ibi_ms)
    x = ibi.copy()
    if len(x) == 0:
        return np.zeros(1, dtype=np.float32), np.zeros(1, dtype=np.float32)
    for i in range(len(x)):
        if not valid[i]:
            left  = max(0, i - 2)
            right = min(len(x), i + 3)
            neigh = x[left:right]
            neigh = neigh[np.isfinite(neigh)]
            if len(neigh) > 0:
                x[i] = np.median(neigh)
            else:
                x[i] = np.median(x[np.isfinite(x)]) if np.isfinite(x).any() else 1.0
    ibi_ms = x * 1000.0
    quality = ((ibi_ms >= min_ibi_ms) & (ibi_ms <= max_ibi_ms)).astype(np.float32)
    return x.astype(np.float32), quality.astype(np.float32)


_N_FEATURES = 10


def _mad_norm(x):
    med = np.median(x)
    mad = np.median(np.abs(x - med)) + 1e-6
    return (x - med) / mad


def build_ibi_features(ibi_sec, quality, max_len_beats=200):
    if len(ibi_sec) == 0:
        return np.zeros((1, _N_FEATURES), dtype=np.float32)

    ibi     = ibi_sec.astype(np.float32)[:max_len_beats]
    quality = quality[:max_len_beats]
    N       = len(ibi)

    ibi_norm      = _mad_norm(ibi)
    dibi          = np.diff(ibi, prepend=ibi[0])
    dibi_norm     = _mad_norm(dibi)
    local_var     = np.array([np.std(ibi[max(0, i-2):min(N, i+3)]) for i in range(N)], dtype=np.float32)
    local_var_norm = _mad_norm(local_var)
    hr            = 60.0 / np.clip(ibi, 0.2, 3.0)
    hr_norm       = _mad_norm(hr)
    abs_dibi      = np.abs(dibi)
    abs_dibi_norm = _mad_norm(abs_dibi)
    pnn50_flag    = (abs_dibi > 0.05).astype(np.float32)
    rel_ibi       = ibi / (np.median(ibi) + 1e-6)
    trend         = np.zeros(N, dtype=np.float32)
    for i in range(N):
        l, r = max(0, i-2), min(N, i+3)
        seg  = ibi[l:r]
        if len(seg) > 1:
            xs = np.arange(len(seg), dtype=np.float32) - (len(seg)-1) / 2.0
            trend[i] = np.dot(xs, seg - seg.mean()) / (np.dot(xs, xs) + 1e-6)
    trend_norm = _mad_norm(trend)
    position   = np.linspace(0.0, 1.0, N, dtype=np.float32)

    return np.stack(
        [ibi_norm, dibi_norm, local_var_norm, quality,
         hr_norm, abs_dibi_norm, pnn50_flag, rel_ibi,
         trend_norm, position],
        axis=-1,
    ).astype(np.float32)


# ─── HRV feature definitions ─────────────────────────────────────────────────

HRV_KEYS = [
    "mean_rr", "sdnn", "rmssd", "pnn50",
    "min_rr", "max_rr", "hr_range",
    "sd1", "sd2", "s_area",
    "lf_power", "hf_power", "lf_hf_ratio", "total_power",
]
N_HRV = len(HRV_KEYS)  # 14

# Indices of features to apply log1p before clipping
_LOG1P_IDX = [1, 2, 7, 8, 9, 10, 11, 13]  # sdnn,rmssd,sd1,sd2,s_area,lf,hf,total


def normalize_hrv(vec: np.ndarray) -> np.ndarray:
    """Robust normalization: log1p on power features, clip all to [-5, 5]."""
    out = vec.astype(np.float64).copy()
    for i in _LOG1P_IDX:
        out[i] = np.log1p(max(0.0, out[i]))
    return np.clip(out, -5.0, 5.0).astype(np.float32)


def compute_hrv_features(
    ibi_sec: np.ndarray,
    fs_interp: float = 4.0,
    lf_band: tuple = (0.04, 0.15),
    hf_band: tuple = (0.15, 0.40),
) -> dict:
    _zero = {k: 0.0 for k in HRV_KEYS}
    ibi = np.asarray(ibi_sec, dtype=np.float64)
    ibi = ibi[np.isfinite(ibi) & (ibi > 0)]
    if len(ibi) < 4:
        return _zero.copy()

    mean_rr  = float(np.mean(ibi))
    sdnn     = float(np.std(ibi, ddof=1)) if len(ibi) > 1 else 0.0
    diff     = np.diff(ibi)
    rmssd    = float(np.sqrt(np.mean(diff ** 2))) if len(diff) > 0 else 0.0
    pnn50    = float(np.mean(np.abs(diff) > 0.05)) if len(diff) > 0 else 0.0
    min_rr   = float(np.min(ibi))
    max_rr   = float(np.max(ibi))
    hr       = 60.0 / np.clip(ibi, 0.2, 3.0)
    hr_range = float(np.max(hr) - np.min(hr))

    x1, x2 = ibi[:-1], ibi[1:]
    sd1     = float(np.std((x2 - x1) / np.sqrt(2), ddof=1)) if len(x1) > 1 else 0.0
    sd2     = float(np.std((x2 + x1) / np.sqrt(2), ddof=1)) if len(x1) > 1 else 0.0
    s_area  = float(np.pi * sd1 * sd2)

    lf_power = hf_power = lf_hf_ratio = total_power = 0.0
    try:
        t_beats   = np.cumsum(np.concatenate([[0.0], ibi[:-1]]))
        t_end     = t_beats[-1]
        if t_end > 0 and len(t_beats) >= 4:
            t_uniform  = np.arange(0.0, t_end, 1.0 / fs_interp)
            kind       = "cubic" if len(t_beats) >= 4 else "linear"
            f_interp   = interp1d(t_beats, ibi, kind=kind,
                                  bounds_error=False, fill_value=(ibi[0], ibi[-1]))
            rr_uniform = f_interp(t_uniform).astype(np.float64)
            nperseg    = min(len(rr_uniform), int(fs_interp * 60))
            if nperseg >= 8:
                freqs, psd  = welch(rr_uniform, fs=fs_interp,
                                    nperseg=nperseg, noverlap=nperseg // 2)
                df          = freqs[1] - freqs[0] if len(freqs) > 1 else 1.0
                lf_mask     = (freqs >= lf_band[0]) & (freqs < lf_band[1])
                hf_mask     = (freqs >= hf_band[0]) & (freqs < hf_band[1])
                lf_power    = float(np.sum(psd[lf_mask]) * df)
                hf_power    = float(np.sum(psd[hf_mask]) * df)
                total_power = float(np.sum(psd) * df)
                lf_hf_ratio = float(lf_power / (hf_power + 1e-10))
    except Exception:
        pass

    return {
        "mean_rr": mean_rr, "sdnn": sdnn, "rmssd": rmssd, "pnn50": pnn50,
        "min_rr": min_rr, "max_rr": max_rr, "hr_range": hr_range,
        "sd1": sd1, "sd2": sd2, "s_area": s_area,
        "lf_power": lf_power, "hf_power": hf_power,
        "lf_hf_ratio": lf_hf_ratio, "total_power": total_power,
    }
