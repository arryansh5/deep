# Dataset-3 Preprocessing & Manifest Report (LDANet Replication)

**Reference Paper:** *“A Lightweight Dual-Branch Multi-Level Attentive Attributes With Constraint Fusion Network for EEG-Based Alzheimer’s Detection”* (IEEE T-ASE 2026)  
**Dataset Source:** Mendeley Data — *Brain Electrophysiological Recording during Olfactory Stimulation in Mild Cognitive Impairment and Alzheimer Disease Patients: An EEG Dataset*

---

## 1. Executive Summary

| Parameter | Value | Notes |
| :--- | :--- | :--- |
| **Cohort Size** | **35 subjects** | 15 CN / Normal, 7 MCI, 13 AD (9 excluded subjects already omitted) |
| **Total Trials** | **3,877 trials** | CN: 1,677 (43.26%), MCI: 848 (21.87%), AD: 1,352 (34.87%) |
| **Channels** | **4 channels** | `Fp1`, `Fz`, `Cz`, `Pz` (Standard 10–20 placement) |
| **Sampling Frequency** | **200 Hz** | 3.0s per trial (-1.0s pre-stimulus to +2.0s post-stimulus) |
| **Samples per Trial** | **600 samples** | Tensor shape: $(4, 600)$ or $(4, 600, 1)$ for 2D convolutions |
| **Normalization** | Channel-wise Z-score | Per-trial standardization ($\mu = 0, \sigma = 1$) |
| **Cross-Validation** | **10-Fold Subject-Stratified** | Zero subject leakage across training and testing folds |

---

## 2. Directory Layout & Artifacts Generated

```text
deep_learning_research/
│
├── data/
│   └── dataset3_processed/
│       ├── CN/                     # 15 subject .npz files (CN_01.npz to CN_15.npz)
│       ├── MCI/                    # 7 subject .npz files (MCI_01.npz to MCI_07.npz)
│       ├── AD/                     # 13 subject .npz files (AD_01.npz to AD_13.npz)
│       ├── X_normed.npy            # Master normalized array (3877, 4, 600)
│       ├── X_raw.npy               # Master unnormalized array (3877, 4, 600)
│       └── y.npy                   # Master ground-truth labels (3877,)
│
├── scripts/
│   ├── preprocess_dataset3.py      # End-to-end processing & manifest builder
│   └── validate_dataset3.py        # Automated integrity & leakage validation
│
├── reports/
│   ├── dataset3_manifest.csv       # Complete trial-level metadata (3,877 rows)
│   ├── dataset3_subject_splits.csv # Subject-to-fold cross-validation partition
│   ├── dataset3_preprocessing_report.json # Machine-readable summary
│   └── dataset3_plots/
│       ├── 01_multichannel_eeg_per_class.png
│       └── 02_raw_vs_normalized_trial.png
```

---

## 3. Visual Diagnostic Inspection

### Multichannel 4-Electrode Olfactory EEG per Class:
![Multichannel 4-Electrode EEG](01_multichannel_eeg_per_class.png)

### Raw vs. Normalized Trial Comparison:
![Raw vs Normalized Trial](02_raw_vs_normalized_trial.png)

---

## 4. Subject-Stratified 10-Fold Partitioning (Zero Leakage)

To reproduce the paper's 10-fold cross-validation results (Section IV-D-5) without trial-level leakage:
- Folds $0 \dots 9$ partition all 35 subjects such that no subject's trials are shared across folds.
- MCI (7 subjects), CN (15 subjects), and AD (13 subjects) are class-stratified.
- Validated with `validate_dataset3.py`: **0 multi-fold subject leaks detected**.
