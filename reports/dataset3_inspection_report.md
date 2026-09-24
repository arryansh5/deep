# Dataset-3 Inspection Report (LDANet Replication)

**Reference Paper:** *“A Lightweight Dual-Branch Multi-Level Attentive Attributes With Constraint Fusion Network for EEG-Based Alzheimer’s Detection”*  
**Dataset Source:** Mendeley Data — *Brain Electrophysiological Recording during Olfactory Stimulation in Mild Cognitive Impairment and Alzheimer Disease Patients: An EEG Dataset*

---

## 1. File Inventory & Storage Layout

- **Directory:** `d:\Projects\deep_learning_research\dataset3\Brain Electrophysiologic\Brain Electrophysiological Recording during Olfactory Stimulation in Mild Cognitive Impairment and Alzheimer Disease Patients An EEG Dataset`
- **Files Found:**
  - `normal.mat` (17,335,985 bytes / ~16.5 MB)
  - `MCI.mat` (7,582,331 bytes / ~7.2 MB)
  - `AD.mat` (14,292,633 bytes / ~13.6 MB)
  - Backup archive: `dataset3/Brain Electrophysiological Recording during Olfactory Stimulation.zip` (39,090,879 bytes)

---

## 2. MATLAB Internal Variable Structure

All three `.mat` files contain a 1D MATLAB struct array of shape `(1, Num_Subjects)` named after the respective diagnosis class:

| File | Top-Level Variable | Struct Dimensions | Internal Struct Fields |
| :--- | :--- | :--- | :--- |
| `normal.mat` | `'normal'` | `(1, 15)` | `('epoch', 'odor', 'noisy')` |
| `MCI.mat` | `'MCI'` | `(1, 7)` | `('epoch', 'odor', 'noisy')` |
| `AD.mat` | `'AD'` | `(1, 13)` | `('epoch', 'odor', 'noisy')` |

### Struct Field Breakdown:
1. **`epoch`**: 3D float array of shape `(4, 600, Num_Trials)`.
   - Dimension 1 (Channels = 4): `Fp1`, `Fz`, `Cz`, `Pz`.
   - Dimension 2 (Time Samples = 600): Captured at 200 Hz spanning 3.0 seconds (1.0s pre-stimulus baseline to 2.0s post-stimulus).
   - Dimension 3 (Trials): Variable number of trials per participant.
2. **`odor`**: 2D uint8 array of shape `(Num_Trials, 1)` containing binary odor stimulus labels (`0` or `1`).
3. **`noisy`**: 2D uint8 array containing annotated trial indices identified as noisy during acquisition.

---

## 3. Cohort & Subject Statistics

| Diagnostic Class | Number of Subjects | Total Trials | Trial Range per Subject | Mean Trials / Subject | Annotated Noisy Trials |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **CN / Normal** | 15 | 1,677 | 44 – 125 | 111.8 | 193 |
| **MCI** | 7 | 848 | 120 – 125 | 121.1 | 16 |
| **AD** | 13 | 1,352 | 59 – 129 | 104.0 | 165 |
| **Total** | **35** | **3,877** | **44 – 129** | **110.8** | **374** |

---

## 4. Signal Integrity & Validation

- **Data Types:** Mixed `float64` and `float32` across subjects.
- **NaN Occurrences:** **0** across all 3,877 trials and 35 subjects.
- **Inf Occurrences:** **0** across all 3,877 trials and 35 subjects.
- **Channel Count:** Strictly **4 channels** across 100% of trials.
- **Trial Length:** Strictly **600 samples** (3.0 seconds at 200 Hz) across 100% of trials.
- **Sampling Frequency:** **200 Hz** (as documented in the original dataset publication and confirmed in LDANet paper Section III-A).

---

## 5. Verification Against the LDANet Paper

| Parameter | Paper Specification | Actual Dataset-3 on Disk | Status |
| :--- | :--- | :--- | :--- |
| **Cohort Size** | 35 participants (15 CN, 7 MCI, 13 AD) | 15 CN, 7 MCI, 13 AD (**35 total**) | **Exact Match** |
| **Excluded Subjects** | 9 excluded from initial 44 | Already excluded in the distributed `.mat` files | **Exact Match** |
| **Channel Count** | 4 channels (`Fp1`, `Fz`, `Cz`, `Pz`) | 4 channels | **Exact Match** |
| **Sampling Rate** | 200 Hz | 200 Hz | **Exact Match** |
| **Trial Tensor Shape** | $4 \times 600 \times \text{Num\_trial}$ | $(4, 600, \text{Num\_trials})$ | **Exact Match** |
| **Trial Duration** | 3.0 seconds (600 samples) | 600 time samples | **Exact Match** |

---

## 6. Preprocessing & Downstream Architectural Considerations

1. **Do NOT apply Dataset-1/Dataset-2 Preprocessing:**
   - Dataset-3 is event-related olfactory EEG (epochs of 600 samples at 200 Hz), unlike continuous resting-state recordings.
   - Preserves trial tensor shape $(4, 600)$ or model input $(4, 600, 1)$.
2. **Subject-Independent Cross-Validation Requirement:**
   - Multiple trials belong to the same subject (e.g., ~110 trials per subject).
   - Splitting must strictly partition subjects across folds, never splitting trials randomly.
3. **Noisy Trials:**
   - All 3,877 trials are currently preserved. An uncleaned and a cleaned metadata flag will be maintained in the manifest.
