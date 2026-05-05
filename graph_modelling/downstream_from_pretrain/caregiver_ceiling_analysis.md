# Caregiver vs Infant Classification — ~63% F1 Ceiling Analysis

## Observed Results

All four classifier heads hit the same ceiling on the caregiver task:

| Head | Test Acc | Test F1 | Kappa |
|------|----------|---------|-------|
| MLP (baseline) | 63.14% | 63.12% | 0.2625 |
| Cosine | 63.45% | 63.44% | 0.2692 |
| Residual MLP | 63.63% | 63.61% | 0.2723 |
| CLS Transformer | 63.02% | 63.02% | 0.2605 |

Compare to sleep/wake on the same encoder and same data: **F1 = 0.876**.

When four completely different architectures converge to the same number, the bottleneck is not the model — it is the task itself.

---

## Root Cause 1: Wrong Signal for This Label

The ECG recording is the **infant's** ECG. The "caregiver" label means a caregiver is
touching or moving the infant — a behavioral event defined from video/annotation files.

The question being posed to the model is:
> *"Can you tell from the infant's heartbeat alone whether someone is currently touching them?"*

This is physiologically hard:
- Autonomic (heart rate) responses to touch are **lagged** — the HR shift takes 5–30 seconds to appear after contact begins.
- The **magnitude** of the response is small compared to sleep/wake transitions.
- The response is **highly variable** across infants and sessions.

Sleep/wake achieves 87% F1 because sleep and wake have persistent, large-magnitude differences in HR, HRV, and beat morphology — the ECG *directly encodes* that physiological state. Caregiving touch has no equivalent direct ECG signature.

---

## Root Cause 2: Behavioral Label vs. Physiological Signal Mismatch

```
Caregiver label  =  INTERACTION annotation (c-active / c-passive / c-pick)
Infant ECG       =  instantaneous cardiac waveform at window start
```

The label boundary is defined by when the caregiver starts or stops moving. The infant's
ECG at second 1 of a caregiving touch looks nearly identical to second 1 of being alone.

The ~63% the model does learn likely comes from **state correlation**: caregivers tend to
touch infants more when they are awake and active, so the model is partially detecting
infant activity state rather than touch itself. This is a confound, not a true signal.

---

## Root Cause 3: 10-Second Windows Are Too Short

A 10-second window captures ~15 beats at infant heart rates (~150 bpm). Any
caregiving-induced HR shift requires at minimum 15–30 seconds to manifest as a
detectable change in the beat sequence. With 10-second windows, the model frequently
sees ECG that precedes the physiological response to the annotated event.

---

## Root Cause 4: High Within-Class Variance

The "infant" and "caregiver" labels each contain very different physiological states:
- **Infant alone, crying** vs. **infant alone, sleeping** → very different ECG
- **Caregiver touching a sleeping infant** vs. **caregiver touching an active infant** → very different ECG

The label does not cleanly partition physiological space, so a large fraction of the
within-class variance overlaps between classes.

---

## Confusion Matrix Pattern

| | Predicted Caregiver | Predicted Infant |
|--|--|--|
| **True Caregiver** (1623) | ~1020 (63%) | ~600 (37%) |
| **True Infant** (1668) | ~610 (37%) | ~1060 (63%) |

Errors are symmetric — ~37% error rate for both classes. The model is not biased or
collapsed; it is genuinely uncertain because the two classes overlap in ECG space.

---

## Improvement Plan

### Short-term (same window-based pipeline)

1. **Longer windows (30 seconds)**
   - Gives the autonomic nervous system time to respond to touch
   - 30s captures ~45 beats, enough to see a meaningful HR trend
   - Change: `--window_sec 30` in the run script

2. **Session-level baseline normalization**
   - Subtract each infant's per-session mean HR and HRV before classification
   - Removes between-infant variance; forces model to detect *relative* changes
   - Implement in the dataset's `__getitem__` by z-scoring beats per session

3. **Temporal context: previous window as baseline**
   - Feed two consecutive windows (pre-event + during-event) to the model
   - Let the model learn the delta rather than the absolute ECG state
   - This directly addresses the lagged response problem

### Medium-term (label/task redesign)

4. **Use transition windows only**
   - Instead of classifying all windows uniformly, focus on windows that straddle
     the annotation boundary (±15 seconds around label onset/offset)
   - These are the windows where the signal is most discriminative

5. **Re-examine label definition**
   - "c-active" (active caregiving) may be more detectable than "c-passive"
   - Splitting caregiver into sub-types and evaluating each separately would
     reveal which interaction types, if any, leave a detectable ECG trace

### Long-term (multi-modal)

6. **Add body movement or audio features**
   - Accelerometer or microphone data alongside ECG would make this task tractable
   - Caregiving touch produces movement artifacts and sound that are directly observable
   - The ECG alone does not have sufficient information to reliably solve this task

---

## Conclusion

The ~63% F1 is likely close to the true **information ceiling** for window-level infant
ECG on this caregiving label definition. Improving the classifier architecture cannot
overcome a fundamental mismatch between the signal (instantaneous infant cardiac activity)
and the label (behavioral event defined by caregiver action). The three highest-leverage
interventions are: longer windows, session normalization, and transition-aware windowing.
