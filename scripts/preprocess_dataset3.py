"""
Preprocessing and Manifest Generation for Dataset-3 (Olfactory EEG Dataset)
Paper: "A Lightweight Dual-Branch Multi-Level Attentive Attributes With Constraint Fusion Network for EEG-Based Alzheimer's Detection" (LDANet)

Dataset-3 Specifications:
- 35 Participants: 15 CN/Normal, 7 MCI, 13 AD
- 4 Channels: Fp1, Fz, Cz, Pz
- Sampling frequency: 200 Hz
- Trial shape: (4, 600) (1.0s pre-stimulus baseline to 2.0s post-stimulus)
- Total Trials: 3,877
- Noisy Annotations: 374 flagged trials
- Subject-Stratified 10-Fold Cross-Validation splits (Zero Subject Leakage)
"""

import os
import glob
import json
import numpy as np
import scipy.io as sio
import pandas as pd
import matplotlib.pyplot as plt

# Directory layout
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RAW_DATA_DIR = os.path.join(
    PROJECT_ROOT, 
    "dataset3", 
    "Brain Electrophysiologic", 
    "Brain Electrophysiological Recording during Olfactory Stimulation in Mild Cognitive Impairment and Alzheimer Disease Patients An EEG Dataset"
)
PROCESSED_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "dataset3_processed")
REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports")
PLOTS_DIR = os.path.join(REPORTS_DIR, "dataset3_plots")

CHANNELS = ["Fp1", "Fz", "Cz", "Pz"]
NUM_CHANNELS = 4
TIME_SAMPLES = 600
FS = 200 # Hz

CLASS_MAPPING = {
    "CN": 0,
    "MCI": 1,
    "AD": 2
}

FILE_CONFIG = [
    {"file": "normal.mat", "var": "normal", "class_name": "CN", "class_label": 0},
    {"file": "MCI.mat", "var": "MCI", "class_name": "MCI", "class_label": 1},
    {"file": "AD.mat", "var": "AD", "class_name": "AD", "class_label": 2},
]

def generate_subject_stratified_folds(seed=42):
    """
    Generates deterministic 10-fold subject-level splits.
    Guarantees no subject appears in both train and validation/test folds.
    """
    rng = np.random.RandomState(seed)
    
    # 15 CN, 7 MCI, 13 AD
    subjects = {
        "CN": [f"CN_{i+1:02d}" for i in range(15)],
        "MCI": [f"MCI_{i+1:02d}" for i in range(7)],
        "AD": [f"AD_{i+1:02d}" for i in range(13)]
    }
    
    subject_fold_map = {}
    
    for cls_name, sub_list in subjects.items():
        shuffled = list(sub_list)
        rng.shuffle(shuffled)
        for idx, sub in enumerate(shuffled):
            fold_idx = idx % 10
            subject_fold_map[sub] = fold_idx
            
    return subject_fold_map

def zscore_trial(trial_arr):
    """
    Applies per-channel Z-score normalization to a single trial of shape (4, 600):
    (x - mean) / std along time axis.
    """
    mean = np.mean(trial_arr, axis=-1, keepdims=True)
    std = np.std(trial_arr, axis=-1, keepdims=True)
    std = np.where(std < 1e-8, 1.0, std)
    return ((trial_arr - mean) / std).astype(np.float32)

def run_pipeline():
    print("=" * 60)
    print("STARTING DATASET-3 PREPROCESSING PIPELINE (LDANET REPLICATION)")
    print(f"Channels: {NUM_CHANNELS} ({', '.join(CHANNELS)})")
    print(f"Sampling Frequency: {FS} Hz")
    print(f"Samples per Trial: {TIME_SAMPLES} (3.0 seconds)")
    print("=" * 60)

    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)
    for cls in CLASS_MAPPING.keys():
        os.makedirs(os.path.join(PROCESSED_DATA_DIR, cls), exist_ok=True)

    subject_fold_map = generate_subject_stratified_folds(seed=42)

    manifest_records = []
    all_raw_trials = []
    all_normed_trials = []
    all_labels = []
    all_subject_ids = []
    
    sample_plots_data = {}
    total_subjects = 0
    total_trials = 0
    total_noisy = 0
    class_trial_counts = {c: 0 for c in CLASS_MAPPING}
    class_subject_counts = {c: 0 for c in CLASS_MAPPING}

    for cfg in FILE_CONFIG:
        fname = cfg["file"]
        varname = cfg["var"]
        cls_name = cfg["class_name"]
        cls_label = cfg["class_label"]
        
        fpath = os.path.join(RAW_DATA_DIR, fname)
        if not os.path.exists(fpath):
            raise FileNotFoundError(f"Missing raw file: {fpath}")
            
        mat = sio.loadmat(fpath)
        struct_arr = mat[varname]
        n_subs = struct_arr.shape[1]
        
        print(f"\nProcessing {cls_name} ({fname}): {n_subs} subjects...")
        
        for sub_idx in range(n_subs):
            total_subjects += 1
            class_subject_counts[cls_name] += 1
            sub_id = f"{cls_name}_{sub_idx+1:02d}"
            fold_idx = subject_fold_map[sub_id]
            
            elem = struct_arr[0, sub_idx]
            epoch_data = elem['epoch'] # (4, 600, Num_trials)
            odor_data = elem['odor']   # (Num_trials, 1) or (Num_trials,)
            noisy_data = elem['noisy'] # (1, N_noisy) or empty
            
            ch, t_len, n_trials = epoch_data.shape
            assert ch == NUM_CHANNELS and t_len == TIME_SAMPLES, f"Shape mismatch: {epoch_data.shape}"
            
            # Extract noisy set (1-based indices in MATLAB)
            noisy_indices = set()
            if noisy_data.size > 0:
                noisy_indices = set(noisy_data.flatten().tolist())
                
            # Process trials for this subject
            sub_raw = []
            sub_normed = []
            
            for tr_idx in range(n_trials):
                total_trials += 1
                class_trial_counts[cls_name] += 1
                
                raw_trial = epoch_data[:, :, tr_idx].astype(np.float32) # (4, 600)
                normed_trial = zscore_trial(raw_trial)                   # (4, 600)
                
                # Check odor
                odor_val = int(odor_data.flat[tr_idx]) if tr_idx < odor_data.size else 0
                
                # Check if trial index is marked noisy (checking both 0-based and 1-based index)
                is_noisy = (tr_idx in noisy_indices) or ((tr_idx + 1) in noisy_indices)
                if is_noisy:
                    total_noisy += 1
                    
                sub_raw.append(raw_trial)
                sub_normed.append(normed_trial)
                
                all_raw_trials.append(raw_trial)
                all_normed_trials.append(normed_trial)
                all_labels.append(cls_label)
                all_subject_ids.append(sub_id)
                
                manifest_records.append({
                    "dataset": "Dataset-3",
                    "subject_id": sub_id,
                    "class_name": cls_name,
                    "class_label": cls_label,
                    "subject_index": sub_idx + 1,
                    "trial_id": f"{sub_id}_T{tr_idx+1:03d}",
                    "original_trial_index": tr_idx,
                    "channels": ",".join(CHANNELS),
                    "channel_count": NUM_CHANNELS,
                    "time_samples": TIME_SAMPLES,
                    "sampling_frequency_hz": FS,
                    "trial_duration_sec": 3.0,
                    "odor_stimulus": odor_val,
                    "is_annotated_noisy": is_noisy,
                    "cv_fold_10": fold_idx,
                    "tensor_shape": f"({NUM_CHANNELS}, {TIME_SAMPLES})"
                })
                
            sub_raw_arr = np.stack(sub_raw, axis=0)       # (N_trials, 4, 600)
            sub_normed_arr = np.stack(sub_normed, axis=0) # (N_trials, 4, 600)
            
            # Save per-subject .npz
            sub_out_path = os.path.join(PROCESSED_DATA_DIR, cls_name, f"{sub_id}.npz")
            np.savez_compressed(
                sub_out_path,
                raw_trials=sub_raw_arr,
                normed_trials=sub_normed_arr,
                class_label=cls_label,
                subject_id=sub_id,
                cv_fold=fold_idx,
                channels=np.array(CHANNELS),
                fs=FS
            )
            
            if cls_name not in sample_plots_data:
                sample_plots_data[cls_name] = {
                    "subject_id": sub_id,
                    "raw": sub_raw_arr[0],
                    "normed": sub_normed_arr[0]
                }

    # Save master arrays for direct model training
    X_raw = np.stack(all_raw_trials, axis=0)       # (3877, 4, 600)
    X_norm = np.stack(all_normed_trials, axis=0)   # (3877, 4, 600)
    y = np.array(all_labels, dtype=np.int64)        # (3877,)
    
    np.save(os.path.join(PROCESSED_DATA_DIR, "X_raw.npy"), X_raw)
    np.save(os.path.join(PROCESSED_DATA_DIR, "X_normed.npy"), X_norm)
    np.save(os.path.join(PROCESSED_DATA_DIR, "y.npy"), y)
    print(f"\nSaved master arrays X (shape {X_norm.shape}) and y (shape {y.shape})")

    # Save manifest CSV
    manifest_df = pd.DataFrame(manifest_records)
    manifest_path = os.path.join(REPORTS_DIR, "dataset3_manifest.csv")
    manifest_df.to_csv(manifest_path, index=False)
    print(f"Saved manifest to: {manifest_path} ({len(manifest_df)} rows)")

    # Save subject split table
    split_df = manifest_df.groupby(["subject_id", "class_name", "cv_fold_10"]).size().reset_index(name="trial_count")
    split_path = os.path.join(REPORTS_DIR, "dataset3_subject_splits.csv")
    split_df.to_csv(split_path, index=False)
    print(f"Saved subject CV splits to: {split_path}")

    # Generate diagnostic plots
    print("Generating visual diagnostic plots...")
    generate_plots(sample_plots_data, PLOTS_DIR)

    # Compile JSON report
    report = {
        "dataset_name": "Dataset-3 (Olfactory Stimulation EEG Dataset)",
        "paper_reference": "A Lightweight Dual-Branch Multi-Level Attentive Attributes With Constraint Fusion Network for EEG-Based Alzheimer's Detection (LDANet)",
        "parameters": {
            "sampling_frequency_hz": FS,
            "channels_count": NUM_CHANNELS,
            "channels_list": CHANNELS,
            "samples_per_trial": TIME_SAMPLES,
            "trial_duration_sec": 3.0,
            "time_window_description": "1.0s pre-stimulus baseline (-1.0s to 0.0s) + 2.0s post-stimulus (0.0s to 2.0s)",
            "normalization_strategy": "Per-trial channel-wise Z-score ((x - mu) / sigma)"
        },
        "subject_statistics": {
            "total_subjects": total_subjects,
            "subjects_per_class": class_subject_counts,
            "cv_strategy": "10-Fold Subject-Stratified Cross-Validation (Zero Subject Leakage)"
        },
        "trial_statistics": {
            "total_trials": total_trials,
            "trials_per_class": class_trial_counts,
            "annotated_noisy_trials": total_noisy,
            "trial_tensor_shape": [NUM_CHANNELS, TIME_SAMPLES],
            "model_input_tensor_shape": [NUM_CHANNELS, TIME_SAMPLES, 1]
        },
        "integrity": {
            "nan_count": int(np.isnan(X_norm).sum()),
            "inf_count": int(np.isinf(X_norm).sum()),
            "mean": float(np.mean(X_norm)),
            "std": float(np.std(X_norm))
        }
    }

    report_json_path = os.path.join(REPORTS_DIR, "dataset3_preprocessing_report.json")
    with open(report_json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"Saved preprocessing report to: {report_json_path}")

    print("\n" + "=" * 60)
    print("DATASET-3 PREPROCESSING & MANIFEST COMPLETED SUCCESSFULLY")
    print(f"Total Subjects: {total_subjects} (CN: {class_subject_counts['CN']}, MCI: {class_subject_counts['MCI']}, AD: {class_subject_counts['AD']})")
    print(f"Total Trials: {total_trials} (CN: {class_trial_counts['CN']}, MCI: {class_trial_counts['MCI']}, AD: {class_trial_counts['AD']})")
    print("=" * 60)
    return report

def generate_plots(sample_plots_data, out_dir):
    """
    Generates high-resolution diagnostic EEG figures for Dataset-3.
    """
    time_axis = np.linspace(-1.0, 2.0, TIME_SAMPLES)
    
    # Plot 1: Representative Multi-Channel EEG Trial for each Class
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    class_order = ["CN", "MCI", "AD"]
    colors = ["#2b5c8f", "#d95f02", "#7570b3"]
    
    for idx, cls in enumerate(class_order):
        if cls in sample_plots_data:
            data = sample_plots_data[cls]["normed"] # (4, 600)
            ax = axes[idx]
            offsets = np.arange(NUM_CHANNELS) * 3.5
            for ch in range(NUM_CHANNELS):
                ax.plot(time_axis, data[ch] + offsets[ch], label=CHANNELS[ch] if idx==0 else "")
            ax.axvline(x=0.0, color='red', linestyle='--', alpha=0.7, label='Odor Onset (t=0s)' if idx==0 else '')
            ax.set_yticks(offsets)
            ax.set_yticklabels(CHANNELS)
            ax.set_title(f"Class: {cls} (Subject: {sample_plots_data[cls]['subject_id']}) - 4-Channel Z-score EEG")
            ax.grid(True, alpha=0.3)
            ax.set_ylabel("Electrode")
            
    axes[0].legend(loc="upper right", ncol=5)
    axes[-1].set_xlabel("Time Relative to Odor Stimulus Onset (seconds)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "01_multichannel_eeg_per_class.png"), dpi=200)
    plt.close()

    # Plot 2: Raw vs Normalized Comparison for a Sample Trial
    if "CN" in sample_plots_data:
        sample = sample_plots_data["CN"]
        fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
        
        for ch in range(NUM_CHANNELS):
            axes[0].plot(time_axis, sample["raw"][ch], label=CHANNELS[ch])
            axes[1].plot(time_axis, sample["normed"][ch], label=CHANNELS[ch])
            
        axes[0].set_title(f"Raw Olfactory EEG Signal (Subject: {sample['subject_id']}, 200 Hz)")
        axes[0].set_ylabel("Raw Amplitude (uV)")
        axes[0].axvline(x=0.0, color='red', linestyle='--', alpha=0.7)
        axes[0].grid(True, alpha=0.3)
        axes[0].legend(loc="upper right", ncol=4)
        
        axes[1].set_title("Z-Score Normalized EEG Signal (mu=0, sigma=1 per channel)")
        axes[1].set_ylabel("Normalized Z-Score")
        axes[1].axvline(x=0.0, color='red', linestyle='--', alpha=0.7)
        axes[1].set_xlabel("Time Relative to Odor Stimulus Onset (seconds)")
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "02_raw_vs_normalized_trial.png"), dpi=200)
        plt.close()

if __name__ == "__main__":
    run_pipeline()
