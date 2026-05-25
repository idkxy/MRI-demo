import argparse
import csv
import os
import time

import numpy as np
import torch

from mri_xai.config import DEVICE, CLASS_NAMES, LABEL_MAP
from mri_xai.model_loader import ModelLoader
from mri_xai.data_processor import DataProcessor
from mri_xai.prediction_pipeline import PredictionPipeline


def get_files_from_dataset(dataset_root, category="All"):
    """
    Collect .npy files from the dataset folder.

    Expected structure:
        dataset_root/
        ├── AD/
        ├── MCI/
        └── CN/
    """

    if category == "All":
        folders = ["AD", "MCI", "CN"]
    else:
        folders = [category]

    all_files = []

    for label in folders:
        folder_path = os.path.join(dataset_root, label)

        if not os.path.exists(folder_path):
            folder_path = os.path.join(dataset_root, label.lower())

        if not os.path.exists(folder_path):
            print(f"Warning: folder not found for {label}: {folder_path}")
            continue

        npy_files = sorted(
            [
                os.path.join(folder_path, f)
                for f in os.listdir(folder_path)
                if f.endswith(".npy")
            ]
        )

        for fpath in npy_files:
            all_files.append((fpath, label))

    return all_files


def compute_top1_slice_fast(cascade_model, raw_volume, true_label_idx):
    """
    Faster top-1 slice importance calculation.

    Instead of doing 30 separate backward passes, this computes gradients
    for all slices in one batch.

    Args:
        cascade_model:
            Single-slice cascade model.

        raw_volume:
            Tensor with shape (S, 3, H, W), values in [0, 1].

        true_label_idx:
            Integer class index: CN=0, MCI=1, AD=2.

    Returns:
        top1_idx, slice_importance
    """

    cascade_model.eval()

    norm_slices = []

    for i in range(raw_volume.shape[0]):
        sl = raw_volume[i].clone().detach().to(DEVICE)
        sl_norm = DataProcessor.imagenet_normalize(sl)
        norm_slices.append(sl_norm)

    # Shape: (S, 3, H, W)
    x = torch.stack(norm_slices).to(DEVICE)
    x = x.clone().detach().requires_grad_(True)

    cascade_model.zero_grad(set_to_none=True)

    # Output shape: (S, 3)
    out = cascade_model(x)

    # Sum target class scores across all slices.
    # Because slices are independent in eval mode, this is equivalent
    # to doing one backward pass per slice, but much faster.
    score = out[:, int(true_label_idx)].sum()

    score.backward()

    grad = x.grad.detach()

    # One importance score per slice.
    slice_importance = (
        grad.flatten(start_dim=1)
        .norm(dim=1)
        .detach()
        .cpu()
        .numpy()
        .astype(np.float32)
    )

    if slice_importance.max() > 0:
        slice_importance = slice_importance / slice_importance.max()

    top1_idx = int(np.argmax(slice_importance))

    return top1_idx, slice_importance


def save_rows_to_csv(rows, output_csv):
    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)

    fieldnames = [
        "file_name",
        "file_path",
        "relative_path",
        "true_label",
        "true_label_idx",
        "volume_prediction",
        "volume_prediction_idx",
        "volume_confidence",
        "volume_cn_prob",
        "volume_mci_prob",
        "volume_ad_prob",
        "top1_slice_index",
        "top1_slice_prediction",
        "top1_slice_prediction_idx",
        "top1_slice_confidence",
        "top1_cn_prob",
        "top1_mci_prob",
        "top1_ad_prob",
        "is_volume_correct",
        "is_top1_correct",
        "processing_time_sec",
    ]

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for row in rows:
            writer.writerow(row)


def save_errors_to_csv(errors, output_csv):
    if not errors:
        return

    error_csv = output_csv.replace(".csv", "_errors.csv")

    fieldnames = [
        "file_name",
        "file_path",
        "true_label",
        "error",
    ]

    with open(error_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for row in errors:
            writer.writerow(row)

    print(f"Saved errors to: {error_csv}")


def precompute_top1_cache(
    dataset_root,
    output_csv,
    category="All",
    max_files=None,
):
    """
    Pre-run top-1 confidence once for every scan in the selected dataset.

    This creates a CSV cache that can later be used by the Streamlit app
    to find best samples quickly.
    """

    if not os.path.exists(dataset_root):
        raise FileNotFoundError(f"Dataset folder not found: {dataset_root}")

    if category not in ["All", "AD", "MCI", "CN"]:
        raise ValueError("category must be one of: All, AD, MCI, CN")

    print("=" * 80)
    print("Top-1 Confidence Precompute")
    print("=" * 80)
    print(f"Device: {DEVICE}")
    print(f"Dataset root: {dataset_root}")
    print(f"Category: {category}")
    print(f"Output CSV: {output_csv}")

    print("\nLoading models...")
    loaded = ModelLoader.load_models()

    stage1 = loaded["stage1"]
    stage2 = loaded["stage2"]
    cascade_model = loaded["cascade_model"]

    stage1.eval()
    stage2.eval()
    cascade_model.eval()

    print(f"Stage 1 checkpoint: {loaded['stage1_checkpoint']}")
    print(f"Stage 2 checkpoint: {loaded['stage2_checkpoint']}")

    files = get_files_from_dataset(dataset_root, category=category)

    if max_files is not None:
        files = files[: int(max_files)]

    total_files = len(files)

    if total_files == 0:
        raise RuntimeError("No .npy files found.")

    print(f"\nFound {total_files} files to process.\n")

    rows = []
    errors = []

    start_all = time.time()

    for idx, (file_path, true_label_name) in enumerate(files, start=1):
        file_start = time.time()

        try:
            true_label_idx = LABEL_MAP[true_label_name]

            # -------------------------------------------------
            # Load scan
            # -------------------------------------------------

            raw_volume = DataProcessor.load_volume(file_path)
            norm_volume = DataProcessor.apply_transforms(raw_volume)

            # -------------------------------------------------
            # Volume-level prediction
            # -------------------------------------------------

            volume_input = norm_volume.unsqueeze(0).to(DEVICE)

            volume_pred_idx, volume_pred_name, volume_probs = (
                PredictionPipeline.cascade_predict_volume(
                    stage1=stage1,
                    stage2=stage2,
                    volume_tensor=volume_input,
                )
            )

            # -------------------------------------------------
            # Fast top-1 slice importance
            # -------------------------------------------------

            top1_idx, slice_importance = compute_top1_slice_fast(
                cascade_model=cascade_model,
                raw_volume=raw_volume,
                true_label_idx=true_label_idx,
            )

            # -------------------------------------------------
            # Top-1 slice prediction
            # -------------------------------------------------

            top1_slice_raw = raw_volume[top1_idx]
            top1_slice_norm = DataProcessor.imagenet_normalize(top1_slice_raw)
            top1_slice_input = top1_slice_norm.unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                top1_out = cascade_model(top1_slice_input)
                top1_probs = top1_out[0].detach().cpu().numpy()

            top1_pred_idx = int(np.argmax(top1_probs))
            top1_pred_name = CLASS_NAMES[top1_pred_idx]
            top1_confidence = float(top1_probs.max())

            processing_time = time.time() - file_start

            row = {
                "file_name": os.path.basename(file_path),
                "file_path": file_path,
                "relative_path": os.path.relpath(file_path, dataset_root),
                "true_label": true_label_name,
                "true_label_idx": int(true_label_idx),
                "volume_prediction": volume_pred_name,
                "volume_prediction_idx": int(volume_pred_idx),
                "volume_confidence": float(volume_probs.max()),
                "volume_cn_prob": float(volume_probs[0]),
                "volume_mci_prob": float(volume_probs[1]),
                "volume_ad_prob": float(volume_probs[2]),
                "top1_slice_index": int(top1_idx),
                "top1_slice_prediction": top1_pred_name,
                "top1_slice_prediction_idx": int(top1_pred_idx),
                "top1_slice_confidence": top1_confidence,
                "top1_cn_prob": float(top1_probs[0]),
                "top1_mci_prob": float(top1_probs[1]),
                "top1_ad_prob": float(top1_probs[2]),
                "is_volume_correct": int(volume_pred_idx == true_label_idx),
                "is_top1_correct": int(top1_pred_idx == true_label_idx),
                "processing_time_sec": round(processing_time, 4),
            }

            rows.append(row)

            elapsed = time.time() - start_all
            avg_time = elapsed / idx
            remaining = avg_time * (total_files - idx)

            print(
                f"[{idx}/{total_files}] "
                f"{os.path.basename(file_path)} | "
                f"true={true_label_name} | "
                f"vol={volume_pred_name} ({volume_probs.max():.3f}) | "
                f"top1={top1_idx} | "
                f"top1_pred={top1_pred_name} ({top1_confidence:.3f}) | "
                f"time={processing_time:.2f}s | "
                f"ETA={remaining / 60:.1f} min",
                flush=True,
            )

        except Exception as e:
            errors.append(
                {
                    "file_name": os.path.basename(file_path),
                    "file_path": file_path,
                    "true_label": true_label_name,
                    "error": str(e),
                }
            )

            print(
                f"[{idx}/{total_files}] ERROR: "
                f"{os.path.basename(file_path)} | {e}",
                flush=True,
            )

    save_rows_to_csv(rows, output_csv)
    save_errors_to_csv(errors, output_csv)

    total_time = time.time() - start_all

    print("\n" + "=" * 80)
    print("Done.")
    print("=" * 80)
    print(f"Processed successfully: {len(rows)}")
    print(f"Errors: {len(errors)}")
    print(f"Total time: {total_time / 60:.2f} min")
    print(f"Saved cache to: {output_csv}")


def main():
    parser = argparse.ArgumentParser(
        description="Precompute top-1 slice confidence for MRI .npy scans."
    )

    parser.add_argument(
        "--dataset_root",
        type=str,
        required=True,
        help="Dataset folder, for example OASIS1, OASIS2, or validation_stage1.",
    )

    parser.add_argument(
        "--output_csv",
        type=str,
        default=None,
        help="Output CSV path. If omitted, uses <dataset_root>_top1_cache.csv.",
    )

    parser.add_argument(
        "--category",
        type=str,
        default="All",
        choices=["All", "AD", "MCI", "CN"],
        help="Category to process.",
    )

    parser.add_argument(
        "--max_files",
        type=int,
        default=None,
        help="Optional maximum number of files to process. Useful for testing.",
    )

    args = parser.parse_args()

    output_csv = args.output_csv

    if output_csv is None:
        clean_name = os.path.basename(os.path.normpath(args.dataset_root))
        output_csv = f"{clean_name}_top1_cache.csv"

    precompute_top1_cache(
        dataset_root=args.dataset_root,
        output_csv=output_csv,
        category=args.category,
        max_files=args.max_files,
    )


if __name__ == "__main__":
    main()