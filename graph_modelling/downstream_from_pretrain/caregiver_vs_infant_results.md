# Caregiver vs Infant Classification — Full Results

## Binary classification results (caregiver vs infant)

| Version | Window | Windowing | Norm | Test Acc | Test F1 | κ | Test N |
|---------|--------|-----------|------|----------|---------|---|--------|
| V1 — MLP | 10 s | all | per-win | 63.14% | 63.12% | 0.263 | 3291 |
| V1 — cosine | 10 s | all | per-win | 63.45% | 63.44% | 0.269 | 3291 |
| V1 — residual | 10 s | all | per-win | 63.63% | 63.61% | 0.272 | 3291 |
| V1 — CLS | 10 s | all | per-win | 63.02% | 63.02% | 0.260 | 3291 |
| V2 — session norm + dual window | 30 s | all | per-session | 69.15% | 69.11% | 0.390 | 953 |
| V3 — transition windows | 30 s | ±15 s boundary | per-win | 71.0% | 70.23% | 0.406 | 262 |
| **V5 — V2 + V3 combined** | 30 s | ±15 s boundary | per-session | **73.7%** | **72.8%** | **0.456** | **262** |

## Sub-type results

| Version | Window | Classes | Test Acc | Macro F1 | κ | Test N |
|---------|--------|---------|----------|---------|---|--------|
| V4 — 3-class sub-types | 30 s | c_active / c_passive / infant | 60.88% | 49.62% | 0.172 | 703 |
| **V6 — c_active vs infant** | 30 s | c_active / infant | **94.6%** | **80.2%** | **0.605** | **552** |

---

## V1 — Four-head plateau

All four heads converge to ~63% on 10-second windows regardless of
architecture. Error rate is symmetric (~37% for both classes).

```
confusion matrix (V1 residual — best head):
               pred caregiver   pred infant
true caregiver      1015            608
true infant          589           1079
```

The model is not biased or collapsed; both classes genuinely overlap in
ECG feature space at 10-second scale. See `caregiver_ceiling_analysis.md`
for the root-cause breakdown.

---

## V2 — Session normalization + dual-window temporal context (+5.5 pp)

**F1: 63.6% → 69.1%** (κ: 0.272 → 0.390)

```
confusion matrix:
               pred caregiver   pred infant
true caregiver      346              92
true infant         202             313
```

Feeding the preceding 30-second window alongside the current one and
normalising with per-session stats lets the model detect the change in
cardiac activity rather than the absolute state. Consistent with the
15–30 s autonomic lag for HR response to touch.

*Note:* 30-second windows produce fewer clips from the same recordings
(953 vs 3291 test samples).

---

## V3 — Transition windows only (+6.6 pp)

**F1: 63.6% → 70.2%** (κ: 0.272 → 0.406)

```
confusion matrix:
               pred caregiver   pred infant
true caregiver      114              44
true infant          32              72
```

Only windows within ±15 s of annotation onset/offset are kept (~511
train, 262 test). Mid-segment steady-state windows are discarded.
The model only ever sees ECG near a state transition where the cardiac
delta is most visible.

*Note:* Small test set (262 samples) — direction is clear but the
exact number has higher variance.

---

## V4 — Sub-type split: c_active / c_passive / infant (3-class)

**Macro F1: 49.62%**, κ = 0.172

```
confusion matrix:
                 pred c_active   pred c_passive   pred infant
true c_active        23               1               13       N=37
true c_passive       27              49               75       N=151
true infant          12             147              356       N=515
```

| Class | Precision | Recall | F1 |
|-------|-----------|--------|----|
| c_active | 37.1% | 62.2% | 46.5% |
| c_passive | 24.9% | 32.5% | 28.2% |
| infant | 80.2% | 69.1% | 74.2% |

**Key finding:** c_active recall (62.2%) is nearly 2× c_passive recall
(32.5%). c_passive and infant are almost indistinguishable (50% of
c_passive windows predicted as infant; 29% of infant windows predicted
as c_passive). This experiment motivated V6.

---

## V5 — V2 + V3 combined: all three improvements (+9.2 pp)

**F1: 63.6% → 72.8%** (κ: 0.272 → 0.456)

```
confusion matrix:
               pred caregiver   pred infant
true caregiver      120              38
true infant          31              73
```

| Class | Precision | Recall | F1 |
|-------|-----------|--------|----|
| caregiver | 79.5% | 75.9% | 77.7% |
| infant | 65.8% | 70.2% | 67.9% |

**Gains are real but diminishing when stacked.** Compared to the two
improvements in isolation:
- V2 standalone (session norm + dual window): +5.5 pp
- V3 standalone (transition windows): +6.6 pp
- V5 combined: +9.2 pp — not fully additive

The incremental gain from adding V2's improvements on top of V3 is
+2.6 pp (72.8% − 70.2%). This is smaller than V2's standalone gain
because transition-window selection already filters to the moments where
the signal is strongest, leaving less room for session normalisation and
dual-window context to add information.

**Kappa of 0.456** (moderate agreement) is the highest of all binary
caregiver vs infant experiments, confirming V5 as the best overall
framing of the binary task.

---

## V6 — c_active vs infant only (cleanest signal)

**Macro F1: 80.2%**, κ = 0.605, acc = 94.6%

```
confusion matrix:
               pred c_active   pred infant
true c_active       26              11        N=37
true infant         19             496        N=515
```

| Class | Precision | Recall | F1 |
|-------|-----------|--------|----|
| c_active | 57.8% | 70.3% | 63.4% |
| infant | 97.8% | 96.3% | 97.1% |

**Accuracy (94.6%) is inflated by the ~14:1 class imbalance** — a model
that always predicts infant would already hit 93.3%. Kappa (0.605,
"substantial agreement") is the correct measure.

**c_active recall of 70.3%** is the highest achieved for active caregiving
detection across all experiments, up from 62.2% in the 3-class V4. Removing
the noisy c_passive class lets the model focus: it no longer needs to
partition feature space for a physiologically ambiguous third class.

**Macro F1 of 80.2%** is not directly comparable to the 63–73% range of the
binary caregiver/infant experiments because the task is different (only
c_active vs infant, c_passive excluded). But it demonstrates that when the
task is restricted to the detectable interaction type, performance jumps
dramatically.

**Class imbalance handling:** Weighted CrossEntropyLoss with
`weight = total / (2 × count)` assigned c_active a weight ≈ 7.5× infant.
No oversampling or architecture change was needed.

---

## Full progression

```
                                              F1      κ      Test N
V1 (10 s, all windows, 4 heads plateau)     63.6%   0.272   3291
  │
  ├─ +5.5 pp ── V2  (30 s, session norm + dual win, all)    69.1%   0.390   953
  │
  ├─ +6.6 pp ── V3  (30 s, transition ±15 s)               70.2%   0.406   262
  │
  └─ +9.2 pp ── V5  (30 s, transition + session + dual)     72.8%   0.456   262
                         ↑ best binary caregiver/infant

V4 (3-class: c_active / c_passive / infant)              49.6%   0.172   703
  └─ V4 diagnostic → c_active recall 62.2% vs c_passive 32.5%
       ↓
V6 (binary: c_active vs infant only)                     80.2%*  0.605   552
                         ↑ best overall, different task
* macro F1; inflated by imbalance relative to accuracy
```

---

## Key conclusions

1. **The ~63% ceiling is not fundamental** — it is an artefact of 10-second
   windows and uniform sampling. With three targeted changes (longer windows,
   transition-aware sampling, temporal context) the binary task reaches 72.8%.

2. **Transition windowing is the highest-leverage single change** (+6.6 pp).
   The cardiac response to touch is not in the steady-state plateau; it is
   at the onset and offset of the interaction. Discarding mid-segment windows
   is more important than any architectural improvement.

3. **Session normalisation and dual-window context are complementary but
   sub-additive.** Together with transition windowing they add +2.6 pp on top
   of V3's +6.6 pp (total +9.2 pp). They are most impactful when transition
   windowing is NOT applied.

4. **Active and passive caregiving are physiologically distinct tasks.** V4
   proved c_passive is nearly indistinguishable from infant. V6 showed that
   when only c_active is used, macro F1 jumps to 80.2% and kappa to 0.605.
   Any future binary caregiver/infant classifier should consider excluding
   c_passive windows from training, or at minimum weighting them down.

5. **The remaining ~30% error on c_active (V6)** is the true near-term
   ceiling. It likely reflects: (a) short-duration active touch segments
   where the cardiac response hasn't manifested yet, (b) high-arousal infant
   states (crying) that mask the touch response, and (c) between-infant
   variability in autonomic reactivity. Session normalisation + transition
   windowing applied to V6 (not yet run) is the natural next experiment.
