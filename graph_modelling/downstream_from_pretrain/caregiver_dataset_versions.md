# Caregiver Dataset Versions

All versions use the same pretrained encoder
(`graph_byol_v4_mse_nodrop/last_checkpoint.pt` unless noted),
the same `residual_mlp` classification head (best performer at baseline),
and the same training hyperparameters (`lr=5e-5`, `batch_size=32`, `epochs=30`,
`freeze_encoder=0`).

---

## V1 — Baseline

**File:** `BRP_dataset_caregiver.py`  
**Script:** `run_sft_caregiver_residual.sh`  
**Output:** `brp_sft_experiments_caregiver/graph_byol_v4_mse_nodrop_residual/`

| Parameter | Value |
|-----------|-------|
| Window size | 10 s |
| max_beats | 32 |
| Normalization | Per-window z-score |
| Windowing | All windows in segment |
| Classes | 2 — `caregiver` / `infant` |
| Encoder ckpt | `last_checkpoint.pt` |

**Results:** acc=63.63%, F1=63.61%, κ=0.272

All four classifier heads (MLP, cosine, residual MLP, CLS transformer) converge
to the same ~63% ceiling, indicating the bottleneck is the task framing rather
than the model. See `caregiver_ceiling_analysis.md` for root-cause analysis.

---

## V1-best — Baseline with best pretrain checkpoint

**File:** `BRP_dataset_caregiver.py` (unchanged)  
**Script:** `run_sft_caregiver_residual_best_ckpt.sh`  
**Output:** `brp_sft_experiments_caregiver/graph_byol_v4_mse_nodrop_residual_best_ckpt/`  
**SLURM job:** 18044625

| Parameter | Value |
|-----------|-------|
| Window size | 10 s |
| max_beats | 32 |
| Normalization | Per-window z-score |
| Windowing | All windows in segment |
| Classes | 2 — `caregiver` / `infant` |
| Encoder ckpt | `best_model.pt` ← changed from V1 |

**Motivation:** The pretrain `best_model.pt` (saved at lowest validation loss
during pretraining) may encode better representations than `last_checkpoint.pt`
(final epoch). All other settings identical to V1.

---

## V1-30s — Longer windows

**File:** `BRP_dataset_caregiver.py` (unchanged)  
**Script:** `run_sft_caregiver_residual_30s.sh`  
**Output:** `brp_sft_experiments_caregiver/graph_byol_v4_mse_nodrop_residual_30s/`  
**SLURM job:** 18044823

| Parameter | Value |
|-----------|-------|
| Window size | **30 s** ← changed |
| max_beats | **80** ← changed |
| Normalization | Per-window z-score |
| Windowing | All windows in segment |
| Classes | 2 — `caregiver` / `infant` |
| Encoder ckpt | `best_model.pt` |

**Motivation (from ceiling analysis §Root Cause 3):**
A 10-second window captures only ~15 infant beats. Autonomic heart-rate
responses to caregiving touch take 15–30 seconds to manifest. 30-second windows
give the model ~75 beats and enough time for a detectable HR change to appear.

`max_beats` must scale with the window: 30 s × 150 bpm ÷ 60 ≈ 75 beats →
`max_beats=80` gives comfortable headroom. The encoder is sequence-length
agnostic (Transformer + GAT with valid_mask), so the pretrained weights load
unchanged.

---

## V2 — Session normalization + dual-window temporal context

**File:** `BRP_dataset_caregiver_v2.py`  
**Script:** `run_sft_caregiver_v2_30s.sh`  
**Output:** `brp_sft_experiments_caregiver/graph_byol_v4_mse_nodrop_v2_30s/`  
**SLURM job:** 18045895

| Parameter | Value |
|-----------|-------|
| Window size | 30 s |
| max_beats per window | 80 |
| Sequence length | **160** (2 × 80, prev + curr concatenated) |
| Normalization | **Per-session z-score** ← changed |
| Windowing | All windows in segment |
| Classes | 2 — `caregiver` / `infant` |
| Encoder ckpt | `last_checkpoint.pt` |

### Change 1 — Session-level normalization

**Why:** Per-window normalization removes amplitude differences within each
10/30-second clip, which is useful for the model but also discards
between-window amplitude variation that may carry state information.
Per-session normalization instead subtracts the whole-recording mean and
divides by the whole-recording std, computed once per ECG file during
`_prepare_index`. This removes between-infant baseline differences (different
sensor placements, body sizes) while preserving relative amplitude changes
across time within the same session.

**Implementation:** `_prepare_index` caches `(mean, std)` per `ecg_path` into
`self.session_stats`. `_extract_window` applies `(segment - mean) / std`
before R-peak detection.

### Change 2 — Dual-window temporal context

**Why:** The model currently classifies each window independently.
The ECG at second 1 of a caregiving touch looks identical to second 1 of
being alone — the delta only becomes visible over time. Feeding the
immediately preceding window alongside the current window lets the model
attend across both and learn the change in cardiac activity rather than
the absolute state.

**Implementation:** `samples` stores `(ecg_path, start, prev_start, label)`.
`__getitem__` calls `_extract_window` twice, pads each to `max_beats`, then
concatenates along the beats dimension:
```
beats      : (2 × max_beats, beat_len)
rr         : (2 × max_beats, 2)
valid_mask : (2 × max_beats,)  — prev slots marked False when no prior data
```
When `prev_start < 0` (first window of a file), the prev half is
zero-filled with `valid_mask=False`. The encoder ignores those positions
via masked attention.

---

## V3 — Transition windows only

**File:** `BRP_dataset_caregiver_v3.py`  
**Script:** `run_sft_caregiver_v3_transition.sh`  
**Output:** `brp_sft_experiments_caregiver/graph_byol_v4_mse_nodrop_v3_transition/`  
**SLURM job:** 18046086

| Parameter | Value |
|-----------|-------|
| Window size | 30 s |
| max_beats | 80 |
| Normalization | Per-window z-score |
| Windowing | **Transition windows only (±15 s of boundary)** ← changed |
| Classes | 2 — `caregiver` / `infant` |
| Encoder ckpt | `last_checkpoint.pt` |

**Why:** Steady-state windows from the middle of a long caregiving segment
carry no more information than baseline infant windows — both look like
normal resting cardiac activity. The discriminative signal lives at the
boundary: windows near the onset of caregiving capture the beginning of the
autonomic HR response, and windows near the offset capture the return to
baseline.

**Implementation:** In `_prepare_index`, for each non-overlapping window `s`
within a segment `[start_sample, end_sample]`:
```python
dist_start = s - start_sample
dist_end   = end_sample - (s + self.window_size)
keep = dist_start <= TRANSITION_SAMPLES or dist_end <= TRANSITION_SAMPLES
```
`TRANSITION_SEC = 15`, `TRANSITION_SAMPLES = 15 000`.
This typically retains the first 0–1 windows and last 0–1 windows of each
segment, discarding the large middle section.

---

## V4 — Caregiver sub-type labels (3-class)

**File:** `BRP_dataset_caregiver_v4.py`  
**Script:** `run_sft_caregiver_v4_subtypes.sh`  
**Output:** `brp_sft_experiments_caregiver/graph_byol_v4_mse_nodrop_v4_subtypes/`  
**SLURM job:** 18046087

| Parameter | Value |
|-----------|-------|
| Window size | 30 s |
| max_beats | 80 |
| Normalization | Per-window z-score |
| Windowing | All windows in segment |
| Classes | **3 — `infant` / `c_active` / `c_passive`** ← changed |
| Encoder ckpt | `last_checkpoint.pt` |

**Why:** The original binary label conflates two physiologically distinct
interactions: c-active (caregiver actively moving/touching the infant) and
c-passive (caregiver's hands resting on the infant without movement).
Active touch produces larger mechanical disturbance, which may elicit a
stronger and faster autonomic response. Splitting them reveals which
sub-type, if any, is actually detectable from infant ECG.

**Label mapping:**

| Annotation tag | V1 label | V4 label |
|----------------|----------|----------|
| `c-active*` | `caregiver` | `c_active` |
| `c-passive*` | `caregiver` | `c_passive` |
| `c-pick*` | `caregiver` | *(excluded)* |
| `i-alone` (crying/active alert, floor) | `infant` | `infant` |
| `i-move` (floor) | `infant` | `infant` |

`c-pick` is excluded: it is a transient spatial movement (infant being lifted),
not a sustained interaction state, and has too few samples to form a reliable
class. With `--num_classes 3` and weighted cross-entropy loss, the model learns
to distinguish active vs passive caregiving from being alone.

**What to look for in results:** If `c_active` F1 >> `c_passive` F1, the
hypothesis in the ceiling analysis (active touch is more detectable) is
confirmed. If both sub-types remain near chance, the bottleneck is fundamental
(the ECG signal simply does not encode either type of touch reliably).

---

## Summary table

| Version | Window | Normalization | Windowing | Classes | Key change |
|---------|--------|---------------|-----------|---------|------------|
| V1 | 10 s | per-window | all | 2 | baseline |
| V1-best | 10 s | per-window | all | 2 | best pretrain ckpt |
| V1-30s | 30 s | per-window | all | 2 | longer window |
| V2 | 30 s | **per-session** | all | 2 | session norm + dual window |
| V3 | 30 s | per-window | **transition ±15 s** | 2 | boundary windows only |
| V4 | 30 s | per-window | all | **3** | c_active / c_passive / infant |
