# EEG Alzheimer's Detection - Preprocessing Pipeline (LDANet Replication)

Replication of the preprocessing pipelines for **Dataset-2** and **Dataset-3** from the research paper:
> **"A Lightweight Dual-Branch Multi-Level Attentive Attributes With Constraint Fusion Network for EEG-Based Alzheimer's Detection"** (IEEE T-ASE 2026)

---

## 1. Project Layout

```text
deep_learning_research/
│
├── data/
│   ├── dataset2_processed/          # Standardized Dataset-2 tensors per subject (166 subjects)
│   │   ├── AD/                      # 57 AD subjects (.npz files)
│   │   ├── MCI/                     # 7 MCI subjects (.npz files)
│   │   └── CONTROL/                 # 102 CONTROL subjects (.npz files)
│   └── dataset3_processed/          # Standardized Dataset-3 tensors (35 subjects, 3,877 trials)
│       ├── AD/                      # 13 AD subjects (.npz files)
│       ├── MCI/                     # 7 MCI subjects (.npz files)
│       ├── CN/                      # 15 CN subjects (.npz files)
│       ├── X_normed.npy             # Master normalized tensor (3877, 4, 600)
│       ├── X_raw.npy                # Master unnormalized tensor (3877, 4, 600)
│       └── y.npy                    # Master labels (3877,)
│
├── scripts/
│   ├── preprocess_dataset2.py       # Dataset-2 preprocessing pipeline
│   ├── validate_dataset2.py         # Dataset-2 integrity & shape validation
│   ├── preprocess_dataset3.py       # Dataset-3 preprocessing pipeline
│   └── validate_dataset3.py         # Dataset-3 integrity & 10-fold CV leakage validation
│
├── reports/
│   ├── dataset2_manifest.csv        # Dataset-2 window-level manifest (25,014 records)
│   ├── dataset2_preprocessing_report.json
│   ├── dataset3_manifest.csv        # Dataset-3 trial-level manifest (3,877 records)
│   ├── dataset3_subject_splits.csv  # Dataset-3 subject-stratified 10-fold CV partitions
│   ├── dataset3_preprocessing_report.json
│   ├── dataset3_preprocessing_report.md
│   ├── plots/                       # Dataset-2 diagnostic figures
│   └── dataset3_plots/              # Dataset-3 diagnostic figures
│
├── A_Lightweight_Dual-Branch_...pdf # Reference research paper
└── README.md
```

---

## 2. Dataset-2 Specifications (Continuous Scalp EEG)

- **Channels:** 19 common standard 10–20 channels (reference electrodes excluded).
- **Sampling Frequency ($f_s$):** Standardized to **256 Hz** (anti-aliased polyphase resampling for 128 Hz recordings).
- **Normalization:** Channel-wise Z-score standardization ($\mu = 0, \sigma = 1$).
- **Window Length:** 4.0 seconds ($1024$ samples).
- **Window Overlap:** 3.0 seconds (75% overlap, $768$ samples, stride $256$).
- **Sample Tensor Output Shape:** `(19, 1024)`
- **Total Windows Generated:** **25,014** (AD: 14,911, CONTROL: 7,491, MCI: 2,612 across 166 usable subjects).

---

## 3. Dataset-3 Specifications (Olfactory Event-Related EEG)

- **Cohort:** 35 subjects (15 CN / Normal, 7 MCI, 13 AD).
- **Channels:** 4 channels (`Fp1`, `Fz`, `Cz`, `Pz`).
- **Sampling Frequency ($f_s$):** **200 Hz**.
- **Trial Duration:** 3.0 seconds (**600 samples** spanning -1.0s pre-stimulus baseline to +2.0s post-stimulus).
- **Trial Tensor Output Shape:** `(4, 600)` (or `(4, 600, 1)` for 2D convolutions).
- **Total Trials Generated:** **3,877** (CN: 1,677, MCI: 848, AD: 1,352).
- **Cross-Validation:** 10-Fold Subject-Stratified partitioning with **zero subject leakage**.

---

## 4. How to Run

### Preprocess Dataset-2:
```bash
python scripts/preprocess_dataset2.py
python scripts/validate_dataset2.py
```

### Preprocess Dataset-3:
```bash
python scripts/preprocess_dataset3.py
python scripts/validate_dataset3.py
```
