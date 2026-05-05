import numpy as np
import warnings
from scipy.signal import butter, filtfilt, find_peaks


def bandpass_filter(ecg, fs):
    nyq = 0.5 * fs
    b, a = butter(2, [5/nyq, 18/nyq], btype='band')
    return filtfilt(b, a, ecg)


def detect_r_peaks(ecg, fs, min_rr_ms=250):
    filtered = bandpass_filter(ecg, fs)
    diff = np.diff(filtered, prepend=filtered[0])
    squared = diff ** 2

    win = int(0.12 * fs)
    mwa = np.convolve(squared, np.ones(win)/win, mode='same')

    distance = int((min_rr_ms / 1000) * fs)
    peaks, _ = find_peaks(mwa, distance=distance, height=np.mean(mwa))

    return peaks


def extract_beats(ecg, r_peaks, fs, max_beats,
                  pre_ratio=0.35, post_ratio=0.50,
                  min_pre_ms=50, min_post_ms=100,
                  max_pre_ms=400, max_post_ms=600):
    """
    Extract beat segments with window size derived from the median RR interval,
    so the window adapts to heart rate and never overlaps adjacent beats.

    All beats in a segment share the same pre/post length (median-RR based),
    which keeps the output array rectangular while still adapting across segments.

    Window derivation:
        pre  = clip(pre_ratio  * median_RR, min_pre_ms,  max_pre_ms)   (ms → samples)
        post = clip(post_ratio * median_RR, min_post_ms, max_post_ms)  (ms → samples)

    Returns
    -------
    beats : np.ndarray  [N, pre+post]  z-score normalised per beat
    rr    : np.ndarray  [N, 2]         [prev_rr, next_rr] in seconds
    """
    min_pre  = int(min_pre_ms  * fs / 1000)
    min_post = int(min_post_ms * fs / 1000)
    max_pre  = int(max_pre_ms  * fs / 1000)
    max_post = int(max_post_ms * fs / 1000)

    if len(r_peaks) < 2:
        warnings.warn(
            "extract_beats: fewer than 2 R-peaks — cannot compute RR. "
            "Returning zero dummy.",
            RuntimeWarning, stacklevel=2,
        )
        return np.zeros((1, min_pre + min_post)), np.zeros((1, 2))

    median_rr = int(np.median(np.diff(r_peaks)))   # samples

    pre  = int(np.clip(pre_ratio  * median_rr, min_pre,  max_pre))
    post = int(np.clip(post_ratio * median_rr, min_post, max_post))

    beats, rr = [], []

    for i in range(1, len(r_peaks) - 1):
        r = r_peaks[i]
        if r - pre < 0 or r + post > len(ecg):
            continue

        beat = ecg[r - pre: r + post]
        beat = (beat - beat.mean()) / (beat.std() + 1e-8)

        prev_rr = (r_peaks[i]     - r_peaks[i - 1]) / fs   # seconds
        next_rr = (r_peaks[i + 1] - r_peaks[i])     / fs

        beats.append(beat)
        rr.append([prev_rr, next_rr])

        if len(beats) >= max_beats:
            break

    if len(beats) == 0:
        warnings.warn(
            f"extract_beats: no valid beats (r_peaks={len(r_peaks)}, "
            f"pre={pre}, post={post}, ecg_len={len(ecg)}). "
            "Returning zero dummy.",
            RuntimeWarning, stacklevel=2,
        )
        return np.zeros((1, pre + post)), np.zeros((1, 2))

    return np.array(beats), np.array(rr)
