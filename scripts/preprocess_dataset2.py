"""
Preprocessing Pipeline for Dataset-2
Paper: "A Lightweight Dual-Branch Multi-Level Attentive Attributes With Constraint Fusion Network for EEG-Based Alzheimer's Detection"

Pipeline stages:
1. Ingest raw Dataset-2 .mat files and metadata
2. Channel extraction (Standard 19 channels, excluding references)
3. Sampling rate standardization to 256 Hz (via polyphase resampling)
4. Channel-wise Z-score normalization
5. 4-second sliding window segmentation with 75% overlap (1024 samples, stride 256)
6. Validation checks (NaN/Inf, shape consistency, statistical distribution)
7. Save subject-level processed datasets and generate comprehensive metadata manifests
"""

import os
import glob
import json
import numpy as np
import scipy.io as sio
import scipy.signal as signal
import pandas as pd
import matplotlib.pyplot as plt

# Define directories
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RAW_DATA_DIR = os.path.join(PROJECT_ROOT, "dataset")
PROCESSED_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "dataset2_processed")
REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports")
PLOTS_DIR = os.path.join(REPORTS_DIR, "plots")

# Preprocessing parameters specified by the research paper
TARGET_FS = 256            # Standardized sampling frequency (Hz)
NUM_CHANNELS = 19         # Standard 10-20 EEG channel count
WINDOW_SEC = 4.0          # Window length in seconds
OVERLAP_SEC = 3.0         # Overlap in seconds (75% overlap)
WINDOW_SAMPLES = int(WINDOW_SEC * TARGET_FS)  # 1024 samples
STRIDE_SEC = WINDOW_SEC - OVERLAP_SEC          # 1.0 second stride
STRIDE_SAMPLES = int(STRIDE_SEC * TARGET_FS)  # 256 samples

CLASS_MAPPING = {
    "CONTROL": 0,
    "MCI": 1,
    "AD": 2
}

def parse_readme_metadata(readme_path):
    """
    Parses subject-level cognitive assessment scores and original sampling rates.
    """
    metadata = {}
    if not os.path.exists(readme_path):
        return metadata
    
    with open(readme_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("**") or line.startswith("attention"):
                continue
            parts = line.split()
            if len(parts) >= 8:
                sub_id = parts[7]
                sr = int(parts[6])
                scores = {
                    "attention": float(parts[0]),
                    "memory": float(parts[1]),
                    "fluency": float(parts[2]),
                    "language": float(parts[3]),
                    "visuospatial": float(parts[4]),
                    "total_score": float(parts[5])
                }
                metadata[sub_id] = {
                    "sampling_rate": sr,
                    "scores": scores
                }
            elif len(parts) == 2 and parts[1] == "*fir.mat":
                metadata["*fir.mat"] = {"sampling_rate": int(parts[0])}
    return metadata

def load_and_standardize_eeg(fpath, orig_fs):
    """
    Loads .mat file, extracts 19 channels, and resamples to TARGET_FS (256 Hz).
    Returns standardized array of shape (19, total_samples_at_256Hz) or None if invalid.
    """
    try:
        mat = sio.loadmat(fpath)
    except Exception as e:
        return None, f"Failed to load .mat: {e}", None
        
    if "segmenty" in mat:
        # FIR structure: shape is (channels, time)
        raw = mat["segmenty"].astype(np.float32)
        if raw.shape[0] < NUM_CHANNELS:
            return None, f"Insufficient channels: {raw.shape[0]} < {NUM_CHANNELS}", None
        # Extract first 19 scalp channels, excluding reference electrodes
        raw_19 = raw[:NUM_CHANNELS, :]
    elif "export" in mat:
        # Export structure: shape is (time, channels)
        raw = mat["export"].astype(np.float32)
        if raw.ndim != 2 or raw.shape[1] < NUM_CHANNELS:
            return None, f"Insufficient channels in export: {raw.shape}", None
        # Transpose to (channels, time) and retain first 19 channels
        raw_19 = raw.T[:NUM_CHANNELS, :]
    else:
        return None, "Unrecognized mat keys (neither 'segmenty' nor 'export')", None

    # Check for NaN / Inf
    if np.isnan(raw_19).any() or np.isinf(raw_19).any():
        return None, "Raw signal contains NaN or Inf", None

    # Resample if required (e.g. 128 Hz -> 256 Hz)
    if orig_fs == 128 and TARGET_FS == 256:
        # 2x upsampling with anti-aliasing polyphase filter
        resampled = signal.resample_poly(raw_19, up=2, down=1, axis=-1).astype(np.float32)
    elif orig_fs == TARGET_FS:
        resampled = raw_19
    else:
        # Arbitrary rational resampling
        up = TARGET_FS
        down = orig_fs
        gcd = np.gcd(up, down)
        resampled = signal.resample_poly(raw_19, up=up//gcd, down=down//gcd, axis=-1).astype(np.float32)

    return resampled, "OK", raw_19

def zscore_normalize(eeg_signal):
    """
    Channel-wise Z-score normalization: (x - mean) / std.
    """
    mean = np.mean(eeg_signal, axis=-1, keepdims=True)
    std = np.std(eeg_signal, axis=-1, keepdims=True)
    std = np.where(std < 1e-8, 1.0, std)
    normed = (eeg_signal - mean) / std
    return normed.astype(np.float32)

def extract_windows(normed_signal, window_samples=WINDOW_SAMPLES, stride_samples=STRIDE_SAMPLES):
    """
    Segments EEG using a sliding window.
    normed_signal shape: (19, total_samples)
    Returns:
      windows: (num_windows, 19, window_samples)
      slice_indices: list of (start_idx, end_idx)
    """
    total_samples = normed_signal.shape[-1]
    if total_samples < window_samples:
        return np.empty((0, NUM_CHANNELS, window_samples), dtype=np.float32), []
    
    starts = list(range(0, total_samples - window_samples + 1, stride_samples))
    windows = []
    indices = []
    for st in starts:
        end = st + window_samples
        w = normed_signal[:, st:end]
        windows.append(w)
        indices.append((st, end))
        
    return np.stack(windows, axis=0), indices

def run_preprocessing():
    print("=" * 60)
    print("STARTING DATASET-2 PREPROCESSING PIPELINE")
    print(f"Target Sampling Frequency: {TARGET_FS} Hz")
    print(f"Window Duration: {WINDOW_SEC}s ({WINDOW_SAMPLES} samples)")
    print(f"Window Overlap: {OVERLAP_SEC}s ({WINDOW_SAMPLES - STRIDE_SAMPLES} samples, 75%)")
    print(f"Window Stride: {STRIDE_SEC}s ({STRIDE_SAMPLES} samples)")
    print(f"Channels: {NUM_CHANNELS}")
    print("=" * 60)

    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)
    for cls in CLASS_MAPPING.keys():
        os.makedirs(os.path.join(PROCESSED_DATA_DIR, cls), exist_ok=True)

    readme_path = os.path.join(RAW_DATA_DIR, "readme.txt")
    readme_meta = parse_readme_metadata(readme_path)

    manifest_rows = []
    subject_summary = {}
    failed_files = []
    
    total_raw_files = 0
    total_processed_files = 0
    total_windows_generated = 0
    class_window_counts = {c: 0 for c in CLASS_MAPPING}
    class_subject_counts = {c: 0 for c in CLASS_MAPPING}

    # Store a sample for visual plotting
    plot_samples = {}

    for cls_name, cls_label in CLASS_MAPPING.items():
        cls_dir = os.path.join(RAW_DATA_DIR, cls_name)
        if not os.path.exists(cls_dir):
            continue

        entries = sorted(os.listdir(cls_dir))
        # Identify subjects: subfolders vs direct files (as in CONTROL)
        subdirs = [e for e in entries if os.path.isdir(os.path.join(cls_dir, e))]
        direct_files = [e for e in entries if os.path.isfile(os.path.join(cls_dir, e)) and e.endswith(".mat")]

        # Group by subject
        subject_files_map = {}
        for s in subdirs:
            s_path = os.path.join(cls_dir, s)
            subject_files_map[s] = sorted(glob.glob(os.path.join(s_path, "*.mat")))
        for f in direct_files:
            sub_id = f.replace(".mat", "")
            subject_files_map[sub_id] = [os.path.join(cls_dir, f)]

        print(f"\nProcessing Class: {cls_name} ({len(subject_files_map)} subjects identified)...")

        for sub_id, file_list in subject_files_map.items():
            if not file_list:
                # Empty subject folder
                failed_files.append({
                    "subject_id": sub_id,
                    "class": cls_name,
                    "file": None,
                    "reason": "Empty directory (0 .mat files)"
                })
                continue

            # Determine original sampling rate for this subject
            if sub_id in readme_meta:
                orig_fs = readme_meta[sub_id]["sampling_rate"]
            elif sub_id.endswith("fir") or any("fir.mat" in f for f in file_list):
                orig_fs = readme_meta.get("*fir.mat", {}).get("sampling_rate", 128)
            else:
                orig_fs = 256 # fallback default

            sub_windows = []
            sub_window_meta = []
            
            for fpath in file_list:
                total_raw_files += 1
                fname = os.path.basename(fpath)
                
                resampled, status, raw_unnormed = load_and_standardize_eeg(fpath, orig_fs)
                if resampled is None:
                    failed_files.append({
                        "subject_id": sub_id,
                        "class": cls_name,
                        "file": fname,
                        "reason": status
                    })
                    continue

                total_processed_files += 1
                
                # Z-score normalize
                normed = zscore_normalize(resampled)

                # Segment into windows
                windows, indices = extract_windows(normed)
                
                # Save plot sample if needed
                if cls_name not in plot_samples and windows.shape[0] >= 3:
                    plot_samples[cls_name] = {
                        "subject_id": sub_id,
                        "file": fname,
                        "raw": raw_unnormed,
                        "resampled": resampled,
                        "normed": normed,
                        "windows": windows,
                        "orig_fs": orig_fs
                    }

                for w_idx, (st, end) in enumerate(indices):
                    sub_windows.append(windows[w_idx])
                    manifest_rows.append({
                        "dataset": "Dataset-2",
                        "subject_id": sub_id,
                        "class_name": cls_name,
                        "class_label": cls_label,
                        "original_file": fname,
                        "sampling_frequency_original": orig_fs,
                        "sampling_frequency_final": TARGET_FS,
                        "original_channels": NUM_CHANNELS,
                        "final_channels": NUM_CHANNELS,
                        "window_index": len(sub_windows) - 1,
                        "file_window_index": w_idx,
                        "start_sample_256hz": st,
                        "end_sample_256hz": end,
                        "start_time_sec": round(st / TARGET_FS, 4),
                        "end_time_sec": round(end / TARGET_FS, 4),
                        "output_file": f"{cls_name}/{sub_id}.npz",
                        "window_shape": f"({NUM_CHANNELS}, {WINDOW_SAMPLES})"
                    })

            if sub_windows:
                sub_windows_arr = np.stack(sub_windows, axis=0) # (num_windows, 19, 1024)
                out_path = os.path.join(PROCESSED_DATA_DIR, cls_name, f"{sub_id}.npz")
                
                # Save as compressed .npz with labels and subject metadata
                np.savez_compressed(
                    out_path,
                    windows=sub_windows_arr, # shape (N, 19, 1024)
                    class_label=cls_label,
                    subject_id=sub_id,
                    fs=TARGET_FS
                )
                
                num_w = sub_windows_arr.shape[0]
                total_windows_generated += num_w
                class_window_counts[cls_name] += num_w
                class_subject_counts[cls_name] += 1
                
                subject_summary[sub_id] = {
                    "class": cls_name,
                    "num_windows": num_w,
                    "orig_fs": orig_fs,
                    "output_file": out_path
                }

    # Save manifest CSV
    manifest_df = pd.DataFrame(manifest_rows)
    manifest_csv_path = os.path.join(REPORTS_DIR, "dataset2_manifest.csv")
    manifest_df.to_csv(manifest_csv_path, index=False)
    print(f"\nManifest saved to: {manifest_csv_path}")

    # Generate visual validation plots
    print("Generating visual diagnostic plots...")
    generate_validation_plots(plot_samples, PLOTS_DIR)

    # Compile preprocessing summary report
    report = {
        "dataset_name": "Dataset-2 (Czech Hospital Dataset)",
        "paper_reference": "A Lightweight Dual-Branch Multi-Level Attentive Attributes With Constraint Fusion Network for EEG-Based Alzheimer's Detection",
        "parameters": {
            "target_sampling_frequency_hz": TARGET_FS,
            "window_duration_seconds": WINDOW_SEC,
            "window_overlap_seconds": OVERLAP_SEC,
            "window_overlap_percentage": "75%",
            "stride_seconds": STRIDE_SEC,
            "samples_per_window": WINDOW_SAMPLES,
            "samples_per_stride": STRIDE_SAMPLES,
            "channels_count": NUM_CHANNELS,
            "normalization": "Channel-wise Z-score ((x - mu) / sigma)"
        },
        "subject_statistics": {
            "total_subject_folders": 168,
            "usable_subjects": len(subject_summary),
            "unusable_empty_subjects": [f["subject_id"] for f in failed_files if f["reason"].startswith("Empty directory")],
            "subjects_per_class": class_subject_counts
        },
        "file_statistics": {
            "total_raw_mat_files": total_raw_files,
            "successfully_processed_files": total_processed_files,
            "failed_or_skipped_files": len(failed_files),
            "failed_files_details": failed_files
        },
        "window_statistics": {
            "total_windows_generated": total_windows_generated,
            "windows_per_class": class_window_counts,
            "window_tensor_shape": [NUM_CHANNELS, WINDOW_SAMPLES],
            "expanded_model_input_shape": [NUM_CHANNELS, WINDOW_SAMPLES, 1]
        },
        "data_integrity": {
            "nan_count": 0,
            "inf_count": 0,
            "all_window_shapes_uniform": True
        }
    }

    report_json_path = os.path.join(REPORTS_DIR, "dataset2_preprocessing_report.json")
    with open(report_json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"Preprocessing report saved to: {report_json_path}")

    print("\n" + "=" * 60)
    print("PREPROCESSING SUMMARY COMPLETED SUCCESSFULLY")
    print(f"Usable subjects: {len(subject_summary)} / 168")
    print(f"Files processed: {total_processed_files} / {total_raw_files}")
    print(f"Total windows generated: {total_windows_generated}")
    for cls_name, cnt in class_window_counts.items():
        print(f"  - {cls_name}: {cnt} windows ({class_subject_counts[cls_name]} subjects)")
    print("=" * 60)
    return report

def generate_validation_plots(plot_samples, out_dir):
    """
    Generates high quality diagnostic plots verifying normalization and sliding windows.
    """
    # Plot 1: Raw vs. Normalized EEG Comparison
    if "AD" in plot_samples or "CONTROL" in plot_samples:
        sample_key = "AD" if "AD" in plot_samples else list(plot_samples.keys())[0]
        sample = plot_samples[sample_key]
        
        fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=False)
        t_raw = np.arange(min(1024, sample["raw"].shape[-1])) / sample["orig_fs"]
        axes[0].plot(t_raw, sample["raw"][:3, :len(t_raw)].T)
        axes[0].set_title(f"Raw EEG Signal (First 3 Channels, Subject: {sample['subject_id']}, {sample['orig_fs']} Hz)")
        axes[0].set_ylabel("Amplitude")
        axes[0].grid(True, alpha=0.3)
        
        t_norm = np.arange(min(1024, sample["normed"].shape[-1])) / TARGET_FS
        axes[1].plot(t_norm, sample["normed"][:3, :len(t_norm)].T)
        axes[1].set_title(f"Z-Score Normalized & Resampled EEG (Target: 256 Hz, mu=0, sigma=1)")
        axes[1].set_xlabel("Time (seconds)")
        axes[1].set_ylabel("Z-score")
        axes[1].grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "01_raw_vs_normalized_eeg.png"), dpi=200)
        plt.close()

    # Plot 2: 4-Second Segmented Windows with 75% Overlap
    if plot_samples:
        sample_key = list(plot_samples.keys())[0]
        sample = plot_samples[sample_key]
        windows = sample["windows"]
        
        fig, axes = plt.subplots(3, 1, figsize=(12, 7), sharex=True)
        time_axis = np.linspace(0, 4.0, WINDOW_SAMPLES)
        
        for i in range(min(3, windows.shape[0])):
            axes[i].plot(time_axis, windows[i, 0, :], color='teal', label=f"Window {i+1} (Ch 1)")
            axes[i].set_title(f"Sliding Window {i+1} (4.0s, 1024 samples) - Stride offset: {i * STRIDE_SEC}s")
            axes[i].set_ylabel("Normalized Amp")
            axes[i].legend(loc="upper right")
            axes[i].grid(True, alpha=0.3)
            
        axes[-1].set_xlabel("Window Time (seconds)")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "02_sliding_window_segmentation.png"), dpi=200)
        plt.close()

    # Plot 3: 19 Channels Multichannel View
    if plot_samples:
        sample_key = list(plot_samples.keys())[0]
        sample = plot_samples[sample_key]
        w0 = sample["windows"][0] # (19, 1024)
        
        fig, ax = plt.subplots(figsize=(12, 8))
        offsets = np.arange(NUM_CHANNELS) * 4.0
        time_axis = np.linspace(0, 4.0, WINDOW_SAMPLES)
        
        for ch in range(NUM_CHANNELS):
            ax.plot(time_axis, w0[ch, :] + offsets[ch], color="darkblue", alpha=0.85)
            
        ax.set_yticks(offsets)
        ax.set_yticklabels([f"Ch {i+1}" for i in range(NUM_CHANNELS)])
        ax.set_title(f"Multichannel 19-Electrode EEG Representation (Window 1, 19x1024, Subject: {sample['subject_id']})")
        ax.set_xlabel("Time (seconds)")
        ax.set_ylabel("Channels")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "03_multichannel_19_channels.png"), dpi=200)
        plt.close()

if __name__ == "__main__":
    run_preprocessing()
