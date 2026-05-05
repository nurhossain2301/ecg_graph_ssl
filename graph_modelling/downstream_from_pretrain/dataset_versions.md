# Caregiver Dataset Versions — Reference

All experiments use the `graph_byol_v4_mse_nodrop` pretrained encoder,
`residual_mlp` head, `lr=5e-5`, `batch_size=32`, `epochs=30`,
`freeze_encoder=0` unless noted.

---

## V1 — Baseline

**File:** `BRP_dataset_caregiver.py`  
**Scripts:** `run_sft_caregiver_residual.sh` (last_ckpt) · `run_sft_caregiver_residual_best_ckpt.sh` (best_ckpt)  
**Output:** `brp_sft_experiments_caregiver/graph_byol_v4_mse_nodrop_residual/`

| | |
|---|---|
| Window | 10 s |
| max_beats | 32 |
| Normalization | Per-window z-score |
| Windowing | All windows in segment |
| Classes | 2 — `caregiver` / `infant` |
| Encoder | `last_checkpoint.pt` |

Baseline. All four heads plateau at ~63% — the signal ceiling at 10 s
window scale. See `caregiver_ceiling_analysis.md` for root-cause breakdown.

---

## V2 — Session normalization + dual-window temporal context

**File:** `BRP_dataset_caregiver_v2.py`  
**Script:** `run_sft_caregiver_v2_30s.sh`  
**SLURM job:** 18045895  
**Output:** `brp_sft_experiments_caregiver/graph_byol_v4_mse_nodrop_v2_30s/`

| | |
|---|---|
| Window | 30 s |
| max_beats | 80 per window → **160 total** (dual) |
| Normalization | **Per-session z-score** |
| Windowing | All windows in segment |
| Classes | 2 — `caregiver` / `infant` |
| Encoder | `last_checkpoint.pt` |

**Change 1 — Session-level normalization:** During `_prepare_index`, the
whole-recording mean and std are computed once per ECG file and cached in
`self.session_stats`. In `_extract_window` the segment is divided by those
stats rather than its own mean/std. This removes between-infant amplitude
bias while preserving relative changes within a session.

**Change 2 — Dual-window context:** Each sample stores
`(ecg_path, start, prev_start, label)`. `__getitem__` calls
`_extract_window` twice — for the preceding window and the current window —
and concatenates their beats:
```
beats       (2 × max_beats, beat_len)
rr          (2 × max_beats, 2)
valid_mask  (2 × max_beats,)   ← prev half masked False when no prior data
```
The encoder (Transformer + GAT) attends across the full 160-beat sequence
and can detect the change in cardiac activity between windows.

---

## V3 — Transition windows only

**File:** `BRP_dataset_caregiver_v3.py`  
**Script:** `run_sft_caregiver_v3_transition.sh`  
**SLURM jobs:** 18046086 (failed, I/O) · 18046277 (failed, I/O) · 18046609 (completed)  
**Output:** `brp_sft_experiments_caregiver/graph_byol_v4_mse_nodrop_v3_transition/`

| | |
|---|---|
| Window | 30 s |
| max_beats | 80 |
| Normalization | Per-window z-score |
| Windowing | **±15 s of annotation onset/offset** |
| Classes | 2 — `caregiver` / `infant` |
| Encoder | `last_checkpoint.pt` |

**Change — Transition windowing:** In `_prepare_index`, for each
non-overlapping window `s` within segment `[start_sample, end_sample]`:

```python
dist_start = s - start_sample
dist_end   = end_sample - (s + window_size)
keep = dist_start <= TRANSITION_SAMPLES or dist_end <= TRANSITION_SAMPLES
# TRANSITION_SAMPLES = 15 * sample_rate = 15 000
```

Mid-segment windows (physiologically at steady-state) are discarded.
Only windows near onset (where the HR response to touch begins) or offset
(where it returns to baseline) are retained. Reduces dataset from ~3291 to
~262 test samples but sharpens the signal.

---

## V4 — Caregiver sub-type labels (3-class)

**File:** `BRP_dataset_caregiver_v4.py`  
**Script:** `run_sft_caregiver_v4_subtypes.sh`  
**SLURM job:** 18046087  
**Output:** `brp_sft_experiments_caregiver/graph_byol_v4_mse_nodrop_v4_subtypes/`

| | |
|---|---|
| Window | 30 s |
| max_beats | 80 |
| Normalization | Per-window z-score |
| Windowing | All windows in segment |
| Classes | **3 — `c_active` / `c_passive` / `infant`** |
| Encoder | `last_checkpoint.pt` |

**Change — Sub-type label split:** `_parse_annotation` maps interaction
tags to three distinct classes rather than collapsing to binary:

| Annotation | V1 label | V4 label |
|------------|----------|----------|
| `c-active*` | `caregiver` | `c_active` |
| `c-passive*` | `caregiver` | `c_passive` |
| `c-pick*` | `caregiver` | *(excluded — transient, too few samples)* |
| `i-alone` (crying/active on floor) | `infant` | `infant` |
| `i-move` (floor) | `infant` | `infant` |

Diagnostic value: reveals that **c_active recall (62.2%) is ~2× c_passive
recall (32.5%)**, confirming active touch leaves a detectable ECG trace
while passive touch is indistinguishable from being alone.

---

## V5 — V2 + V3 combined (all three improvements)

**File:** `BRP_dataset_caregiver_v5.py`  
**Script:** `run_sft_caregiver_v5_combined.sh`  
**SLURM job:** 18047002  
**Output:** `brp_sft_experiments_caregiver/graph_byol_v4_mse_nodrop_v5_combined/`

| | |
|---|---|
| Window | 30 s |
| max_beats | 80 per window → **160 total** (dual) |
| Normalization | **Per-session z-score** |
| Windowing | **±15 s of annotation onset/offset** |
| Classes | 2 — `caregiver` / `infant` |
| Encoder | `last_checkpoint.pt` |

Applies all three improvements simultaneously:
1. Transition windows only (from V3)
2. Session-level normalization (from V2)
3. Dual-window temporal context (from V2)

Tests whether the gains from V2 (+5.5 pp) and V3 (+6.6 pp) are additive.

---

## V6 — c_active vs infant only (cleanest signal, imbalanced)

**File:** `BRP_dataset_caregiver_v6.py`  
**Script:** `run_sft_caregiver_v6_cactive.sh`  
**SLURM job:** 18047003  
**Output:** `brp_sft_experiments_caregiver/graph_byol_v4_mse_nodrop_v6_cactive/`

| | |
|---|---|
| Window | 30 s |
| max_beats | 80 |
| Normalization | Per-window z-score |
| Windowing | All windows in segment |
| Classes | **2 — `c_active` / `infant`** |
| Encoder | `last_checkpoint.pt` |
| Class imbalance | ~14:1 (infant >> c_active) |

**Change — Drop c_passive and c_pick:** `_parse_annotation` only emits
`c_active` (for `c-active*`) and `infant` (for `i-alone` / `i-move`).
`c-passive*` and `c-pick*` are ignored entirely.

**Imbalance handling:** `_prepare_index` logs class counts on startup.
The existing `compute_class_weights` in `main.py` assigns
`weight = total / (num_classes × count)` per class, so `c_active` gets
a weight ~14× higher than `infant`. Weighted `CrossEntropyLoss` in
`train.py` applies these at every step — no architecture change needed.

Motivation: V4 showed c_active is the only interaction type with a
detectable ECG trace. Isolating it from c_passive (which adds noise)
should push binary F1 above the ~70% reached by V3.

---

## Comparison table

| Version | Window | Norm | Windowing | Classes | Result |
|---------|--------|------|-----------|---------|--------|
| V1 | 10 s | per-win | all | caregiver/infant | F1 63.6% |
| V2 | 30 s | **per-session** | all + **dual-win** | caregiver/infant | F1 69.1% |
| V3 | 30 s | per-win | **transition ±15 s** | caregiver/infant | F1 70.2% |
| V4 | 30 s | per-win | all | **c_active/c_passive/infant** | F1 49.6% (3-class) |
| V5 | 30 s | **per-session** | **transition ±15 s** + **dual-win** | caregiver/infant | pending |
| V6 | 30 s | per-win | all | **c_active/infant** | pending |
