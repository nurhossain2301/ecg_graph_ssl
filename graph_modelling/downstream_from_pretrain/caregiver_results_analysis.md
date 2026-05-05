# Caregiver vs Infant — Results Analysis

## Results Summary

| Version | Change | Test Acc | Test F1 | κ | Test N |
|---------|--------|----------|---------|---|--------|
| V1 — baseline (MLP) | 10 s, all windows | 63.14% | 63.12% | 0.263 | 3291 |
| V1 — cosine head | 10 s, all windows | 63.45% | 63.44% | 0.269 | 3291 |
| V1 — residual head | 10 s, all windows | 63.63% | 63.61% | 0.272 | 3291 |
| V1 — CLS head | 10 s, all windows | 63.02% | 63.02% | 0.260 | 3291 |
| **V2 — session norm + dual window** | 30 s, session norm, prev+curr windows | **69.15%** | **69.11%** | **0.390** | 953 |
| **V3 — transition windows only** | 30 s, ±15 s of boundary | **71.0%** | **70.23%** | **0.406** | 262 |
| V4 — sub-type labels | 30 s, 3-class (c_active / c_passive / infant) | 60.88% | 49.62% | 0.172 | 703 |

---

## Finding 1 — Transition windows are the single most effective change

V3 (transition windows ±15 s of annotation onset/offset) reaches **F1 = 70.2%**,
the highest of all binary experiments, compared to the V1 ceiling of 63.6%.
That is a **+6.6 pp absolute improvement** with no architecture change — just a
different data-sampling strategy.

This directly validates Root Cause 3 from the ceiling analysis: steady-state
windows from the middle of a long caregiving segment are near-uninformative
because the infant's ECG has already settled into whatever state the touch
induces. Windows near the boundary capture the cardiac response *in transition*,
which is the most discriminative moment.

**Caveat — small test set:** The V3 test split has only 262 samples (vs 3291 for V1)
because transition sampling drastically reduces the number of windows per session.
The F1 estimate has higher variance. The directional finding is clear, but the
exact number should be interpreted with caution.

---

## Finding 2 — Session normalization + dual-window context helps substantially

V2 reaches **F1 = 69.1%** (+5.5 pp over V1), using:
- **Session-level normalization**: removes between-infant amplitude bias so the
  model detects *relative* changes rather than absolute signal magnitude.
- **Dual-window (previous + current)**: the model attends across 160 beats
  (preceding 30 s + current 30 s) and can learn the delta in cardiac activity.

The improvement is consistent with the ceiling analysis prediction: the
physiological response to touch takes 15–30 s to manifest, so giving the model
the baseline window makes the change detectable even when the current window
alone is ambiguous.

**Note on sample count:** 30 s windows yield fewer non-overlapping windows from
the same recordings (953 vs 3291 test samples). Part of the gain may reflect
a harder but cleaner evaluation. Regardless, the kappa improvement (0.272 →
0.390) is large.

---

## Finding 3 — Active vs passive caregiving are not equally detectable

V4 splits the `caregiver` class into `c_active` and `c_passive`:

### Confusion matrix
```
                 Predicted
                 c_active  c_passive  infant
True  c_active  [  23         1         13  ]   N=37
      c_passive [  27        49         75  ]   N=151
      infant    [  12       147        356  ]   N=515
```

### Per-class performance
| Class | Precision | Recall | F1 |
|-------|-----------|--------|----|
| c_active | 37.1% (23/62) | 62.2% (23/37) | 46.5% |
| c_passive | 24.9% (49/197) | 32.5% (49/151) | 28.2% |
| infant | 80.2% (356/444) | 69.1% (356/515) | 74.2% |

**c_active recall (62.2%) is nearly 2× c_passive recall (32.5%).**
This confirms the hypothesis from the ceiling analysis: active caregiving
touch leaves a more detectable cardiac trace in the infant than passive
caregiving. The model can distinguish when a caregiver is actively
touching/moving the infant more reliably than when they are passively
resting hands on the infant.

**c_passive vs infant confusion:** 147 out of 515 infant windows are
predicted as c_passive (29%), and 75 out of 151 c_passive windows are
predicted as infant (50%). The ECG during passive caregiving is nearly
indistinguishable from the ECG when the infant is alone — this is consistent
with the physiological argument that passive touch produces little autonomic
response.

**Macro F1 drops to 49.6%** because the harder 3-class problem introduces
the very-low-F1 c_passive class into the average. Accuracy (60.9%) stays
near the V1 baseline only because the model is correct on infant most of
the time. This version is most useful for its diagnostic value rather than
as a deployment-ready classifier.

---

## Overall progression

```
V1 baseline (all heads plateau at ~63%)
    │
    ├── +5.5 pp ── V2: longer window + session norm + dual window  → F1 69.1%
    │
    └── +6.6 pp ── V3: transition windows only                     → F1 70.2%
```

The two most impactful changes are independent and potentially additive:
**V3 + V2 improvements combined** (transition windows AND session normalization
AND dual-window) has not been run yet and is the natural next experiment.

---

## What the sub-type analysis tells us about the task

The V4 confusion matrix has a strong structural message:

- **infant ↔ c_passive** are mutually confused (the largest off-diagonal
  cells in the matrix). This means passive caregiving produces ECG that is
  physiologically equivalent to being alone — the infant's autonomic system
  does not register a detectable response to passive touch at the 30-second
  window scale.

- **c_active is more separable** from infant (62% recall), confirming that
  active touch does produce a small but real cardiac response.

- **c_active ↔ c_passive** are also confused (27 c_passive predicted as
  c_active). This likely reflects within-session transitions where the
  caregiver shifts between active and passive contact.

**Practical implication:** A binary classifier that collapses c_active and
c_passive into one `caregiver` class is mixing a detectable signal (active
touch) with noise (passive touch). A future experiment should test a binary
classifier trained only on c_active vs infant, dropping c_passive entirely.
This would present the model with the cleanest possible version of the task.

---

## Recommended next experiments

| Priority | Experiment | Expected outcome |
|----------|-----------|-----------------|
| High | V3 + V2: transition windows + session norm + dual window | Potentially additive gains, targeting 72–74% F1 |
| High | Binary c_active vs infant only (drop c_passive) | Cleaner signal; c_active alone may reach 70–75% binary F1 |
| Medium | V3 transition sampling applied to the V4 sub-type split | Test whether transition windows help c_active detection specifically |
| Low | Ensembling V2 + V3 predictions on overlapping test samples | Small practical gain |
