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


def build_ibi_features(ibi_sec, quality, max_len_beats=128):
    """
    Node features per beat interval:
      0: normalized IBI
      1: normalized delta-IBI
      2: local variability (rolling std)
      3: quality
    """
    if len(ibi_sec) == 0:
        feats = np.zeros((1, 4), dtype=np.float32)
        return feats

    ibi = ibi_sec.astype(np.float32)

    # truncate
    ibi = ibi[:max_len_beats]
    quality = quality[:max_len_beats]

    med = np.median(ibi)
    mad = np.median(np.abs(ibi - med)) + 1e-6

    ibi_norm = (ibi - med) / mad

    dibi = np.diff(ibi, prepend=ibi[0])
    dmed = np.median(dibi)
    dmad = np.median(np.abs(dibi - dmed)) + 1e-6
    dibi_norm = (dibi - dmed) / dmad

    local_var = np.zeros_like(ibi)
    for i in range(len(ibi)):
        l = max(0, i - 2)
        r = min(len(ibi), i + 3)
        local_var[i] = np.std(ibi[l:r])

    lv_med = np.median(local_var)
    lv_mad = np.median(np.abs(local_var - lv_med)) + 1e-6
    local_var_norm = (local_var - lv_med) / lv_mad

    feats = np.stack(
        [ibi_norm, dibi_norm, local_var_norm, quality],
        axis=-1
    ).astype(np.float32)

    return feats


def random_window_ibi(ibi, window_beats=128, train=True):
    if len(ibi) <= window_beats:
        return ibi

    if train:
        start = np.random.randint(0, len(ibi) - window_beats + 1)
    else:
        start = max(0, (len(ibi) - window_beats) // 2)

    return ibi[start:start + window_beats]