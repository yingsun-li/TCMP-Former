# TCMP-Former

**A Framework Combining Temporal Context and Morphological Patch Representations for Seizure-Onset Pattern Recognition in Intracranial EEG**

TCMP-Former (Temporal Context–Morphology Patch Transformer) is a framework for detecting and classifying seizure-onset patterns (SOPs) in intracranial EEG. It combines fine-grained waveform morphology, local statistical scale, continuous waveform structure, and temporal context to characterize seizure-onset activity.

## Method overview

- **Morphology-factorized patch encoding (MFPE)** represents local waveform shape together with mean- and RMS-based statistical conditioning.
- **Temporal CNN encoding** captures neighboring continuous waveform structure.
- **CNN-guided local cross-attention** fuses morphological representations with continuous-structure features.
- **Temporal-context modeling** relates preceding activity to the current signal representation for SOP recognition.

## Study scope

The study investigates:

- Binary SOP recognition and eight-class morphological classification.
- Comparisons with baseline methods and component ablation analyses.
- Onset-aligned model responses, seizure-level onset timing, and matched interictal comparisons.
- Target-center adaptation through training from scratch, fine-tuning, and joint training.

## Data

The study uses the public HUP iEEG Epilepsy Dataset and a non-public clinical cohort from Xuanwu Hospital.

- **Public HUP dataset:** available through [OpenNeuro, ds004100](https://openneuro.org/datasets/ds004100).
- **Xuanwu Hospital cohort:** not publicly available because of patient privacy restrictions. Access requests may be directed to the corresponding authors and are subject to institutional approval.

## Code availability

**Full implementation and reproduction instructions: coming soon.**
