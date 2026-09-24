# EEG Alzheimer's Detection - Preprocessing Pipeline (Dataset-2)

Implementation of the preprocessing stage for **Dataset-2** from the paper:
> **"A Lightweight Dual-Branch Multi-Level Attentive Attributes With Constraint Fusion Network for EEG-Based Alzheimer's Detection"**

---

## 1. Project Layout

```text
deep_learning_research/
│
├── data/
│   └── dataset2_processed/          # Standardized preprocessed tensors grouped per subject
│       ├── AD/                      # 57 AD subjects (.npz files)
│       ├── MCI/                     # 7 MCI subjects (.npz files)
│       └── CONTROL/                 # 102 CONTROL subjects (.npz files)
│
├── dataset/                         # Original raw dataset (UNMODIFIED)
│   ├── AD/
│   ├── CONTROL/
│   ├── MCI/
│   └── readme.txt
│
├── scripts/
│   ├── preprocess_dataset2.py       # Full preprocessing execution pipeline
│   └── validate_dataset2.py         # Rigorous data integrity & shape validation script
│
├── reports/
│   ├── dataset2_manifest.csv        # Traceable window-level manifest (25,014 records)
│   ├── dataset2_preprocessing_report.json # Comprehensive JSON summary report
│   └── plots/                       # Diagnostic visual verification figures
│       ├── 01_raw_vs_normalized_eeg.png
│       ├── 02_sliding_window_segmentation.png
│       └── 03_multichannel_19_channels.png
│
├── A_Lightweight_Dual-Branch_...pdf # Reference research paper
└── README.md
```

---

## 2. Preprocessing Specifications

- **Channels ($e$):** 19 common standard 10–20 channels (reference electrodes excluded).
- **Sampling Frequency ($f_s$):** Standardized to **256 Hz** (anti-aliased polyphase resampling for 128 Hz recordings).
- **Normalization:** Channel-wise Z-Score standardization ($\mu = 0, \sigma = 1$).
- **Window Length ($\Delta t_w$):** 4.0 seconds ($L_w = 4 \times 256 = 1024$ samples).
- **Window Overlap ($\Delta t_o$):** 3.0 seconds (75% overlap, 768 samples).
- **Window Stride ($L_s$):** 1.0 second ($256$ samples).
- **Sample Tensor Output Shape:** `(19, 1024)` (or `(19, 1024, 1)` with model channel dimension).

---

## 3. Dataset-2 Summary

- **Total Usable Subjects:** 166 (CONTROL: 102, MCI: 7, AD: 57). Note: folders `AD33` and `AD44` are empty in the raw distribution.
- **Total Processed Files:** 464 `.mat` files.
- **Total Windows Generated:** **25,014**
  - **AD:** 14,911 windows (59.61%)
  - **CONTROL:** 7,491 windows (29.95%)
  - **MCI:** 2,612 windows (10.44%)
- **Data Leakage Isolation:** All windows retain their `subject_id` and `class_label` metadata for subject-stratified cross-validation.

---

## 4. How to Run

### Execute Preprocessing:
```bash
python scripts/preprocess_dataset2.py
```

### Validate Integrity:
```bash
python scripts/validate_dataset2.py
```
