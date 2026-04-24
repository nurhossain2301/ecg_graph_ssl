import numpy as np
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

    distance = int((min_rr_ms/1000)*fs)
    peaks, _ = find_peaks(mwa, distance=distance, height=np.mean(mwa))

    return peaks

def extract_beats(ecg, r_peaks, fs, pre_ms, post_ms, max_beats):
    pre = int(pre_ms/1000 * fs)
    post = int(post_ms/1000 * fs)

    beats, rr = [], []

    for i in range(1, len(r_peaks)-1):
        r = r_peaks[i]
        if r-pre < 0 or r+post > len(ecg):
            continue

        beat = ecg[r-pre:r+post]
        beat = (beat - beat.mean()) / (beat.std() + 1e-8)

        prev_rr = (r_peaks[i] - r_peaks[i-1]) / fs
        next_rr = (r_peaks[i+1] - r_peaks[i]) / fs

        beats.append(beat)
        rr.append([prev_rr, next_rr])

        if len(beats) >= max_beats:
            break

    if len(beats) == 0:
        return np.zeros((1, pre+post)), np.zeros((1,2))

    return np.array(beats), np.array(rr)