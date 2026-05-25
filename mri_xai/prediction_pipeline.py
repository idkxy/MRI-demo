import os
import csv
import numpy as np
import torch

from mri_xai.config import DEVICE, CLASS_NAMES, LABEL_MAP
from mri_xai.model_loader import ModelLoader
from mri_xai.data_processor import DataProcessor
from mri_xai.xai_methods import XAIMethods
from mri_xai.plotter import XAIPlotter


class PredictionPipeline:
    @staticmethod
    def cascade_predict_volume(stage1, stage2, volume_tensor):
        """
        Run the full two-stage cascade on a full scan.

        Stage 1:
            CN vs Impaired

        Stage 2:
            MCI vs AD, only used when Stage 1 predicts impaired.
        """

        with torch.no_grad():
            s1_out = stage1(volume_tensor)
            s1_prob = torch.softmax(s1_out, dim=1)

            s1_pred = s1_prob.argmax(1).item()

            if s1_pred == 0:
                p_cn = s1_prob[0, 0].item()
                p_imp = s1_prob[0, 1].item()

                probs = np.array([
                    p_cn,
                    p_imp * 0.5,
                    p_imp * 0.5,
                ])

                pred_idx = int(np.argmax(probs))
                pred_name = CLASS_NAMES[pred_idx]

                return pred_idx, pred_name, probs

            s2_out = stage2(volume_tensor)
            s2_prob = torch.softmax(s2_out, dim=1)

            p_cn = s1_prob[0, 0].item()
            p_mci = s1_prob[0, 1].item() * s2_prob[0, 0].item()
            p_ad = s1_prob[0, 1].item() * s2_prob[0, 1].item()

            probs = np.array([p_cn, p_mci, p_ad])

            pred_idx = int(np.argmax(probs))
            pred_name = CLASS_NAMES[pred_idx]

            return pred_idx, pred_name, probs

    @staticmethod
    def predict_single_slice(cascade_model, slice_tensor):
        """
        Run the single-slice cascade model.

        Args:
            cascade_model:
                Single-slice cascade model.

            slice_tensor:
                Tensor with shape (1, 3, H, W)

        Returns:
            pred_idx, pred_name, probs
        """

        with torch.no_grad():
            out = cascade_model(slice_tensor)
            probs = out[0].detach().cpu().numpy()

        pred_idx = int(np.argmax(probs))
        pred_name = CLASS_NAMES[pred_idx]

        return pred_idx, pred_name, probs

    @staticmethod
    def build_top1_cache(
        validation_root,
        cache_path,
        category="All",
        max_files_per_class=None,
    ):
        """
        Pre-compute top-1 slice confidence for scans and save to CSV.

        This is the slow step. After this cache is created, best-sample search
        can simply read from the CSV file instead of re-running the model.

        Args:
            validation_root:
                Dataset folder containing AD, MCI, CN folders.

            cache_path:
                Output CSV path.

            category:
                "AD", "MCI", "CN", or "All".

            max_files_per_class:
                Optional limit for testing. Use None to process all files.

        Returns:
            {
                "cache_path": cache_path,
                "num_rows": number of successfully processed scans,
                "errors": [...]
            }
        """

        if not os.path.exists(validation_root):
            raise FileNotFoundError(
                f"Validation folder not found: {validation_root}"
            )

        if category not in ["AD", "MCI", "CN", "All"]:
            raise ValueError(
                "category must be one of: AD, MCI, CN, All"
            )

        loaded = ModelLoader.load_models()

        stage1 = loaded["stage1"]
        stage2 = loaded["stage2"]
        cascade_model = loaded["cascade_model"]

        if category == "All":
            folder_order = ["AD", "MCI", "CN"]
        else:
            folder_order = [category]

        rows = []
        errors = []

        for folder in folder_order:
            folder_path = os.path.join(validation_root, folder)

            if not os.path.exists(folder_path):
                folder_path = os.path.join(validation_root, folder.lower())

            if not os.path.exists(folder_path):
                errors.append(
                    {
                        "file_name": "",
                        "file_path": folder_path,
                        "true_label": folder,
                        "error": f"Folder not found for class {folder}",
                    }
                )
                continue

            true_label_name = folder
            true_label_idx = LABEL_MAP[true_label_name]

            npy_files = sorted(
                [
                    os.path.join(folder_path, f)
                    for f in os.listdir(folder_path)
                    if f.endswith(".npy")
                ]
            )

            if max_files_per_class is not None:
                npy_files = npy_files[: int(max_files_per_class)]

            for file_path in npy_files:
                try:
                    raw_volume = DataProcessor.load_volume(file_path)
                    norm_volume = DataProcessor.apply_transforms(raw_volume)

                    volume_input = norm_volume.unsqueeze(0).to(DEVICE)

                    volume_pred_idx, volume_pred_name, volume_probs = (
                        PredictionPipeline.cascade_predict_volume(
                            stage1=stage1,
                            stage2=stage2,
                            volume_tensor=volume_input,
                        )
                    )

                    slice_importance, top1_idx = XAIMethods.compute_slice_importance(
                        cascade_model=cascade_model,
                        raw_volume=raw_volume,
                        target_class=int(true_label_idx),
                    )

                    top1_slice_raw = raw_volume[top1_idx]
                    top1_slice_norm = DataProcessor.imagenet_normalize(top1_slice_raw)
                    top1_slice_input = top1_slice_norm.unsqueeze(0).to(DEVICE)

                    top1_pred_idx, top1_pred_name, top1_probs = (
                        PredictionPipeline.predict_single_slice(
                            cascade_model=cascade_model,
                            slice_tensor=top1_slice_input,
                        )
                    )

                    top1_confidence = float(top1_probs.max())

                    rows.append(
                        {
                            "file_name": os.path.basename(file_path),
                            "file_path": file_path,
                            "relative_path": os.path.relpath(
                                file_path,
                                validation_root,
                            ),
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
                        }
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

        os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)

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
        ]

        with open(cache_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for row in rows:
                writer.writerow(row)

        return {
            "cache_path": cache_path,
            "num_rows": len(rows),
            "errors": errors,
        }

    @staticmethod
    def load_top1_cache(cache_path):
        """
        Load the precomputed top-1 confidence cache.
        """

        if not os.path.exists(cache_path):
            return []

        rows = []

        with open(cache_path, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)

            for row in reader:
                parsed = dict(row)

                int_fields = [
                    "true_label_idx",
                    "volume_prediction_idx",
                    "top1_slice_index",
                    "top1_slice_prediction_idx",
                    "is_volume_correct",
                    "is_top1_correct",
                ]

                float_fields = [
                    "volume_confidence",
                    "volume_cn_prob",
                    "volume_mci_prob",
                    "volume_ad_prob",
                    "top1_slice_confidence",
                    "top1_cn_prob",
                    "top1_mci_prob",
                    "top1_ad_prob",
                ]

                for key in int_fields:
                    parsed[key] = int(float(parsed[key]))

                for key in float_fields:
                    parsed[key] = float(parsed[key])

                rows.append(parsed)

        return rows

    @staticmethod
    def find_best_samples_from_cache(
        cache_path,
        min_confidence=0.80,
        max_samples=10,
        category="All",
    ):
        """
        Fast best-sample search using the precomputed CSV cache.

        A best sample must satisfy:
            1. Volume prediction is correct.
            2. Top-1 slice prediction is correct.
            3. Top-1 slice confidence >= min_confidence.

        If category == "All", results are balanced across AD, MCI, and CN.
        """

        if category not in ["AD", "MCI", "CN", "All"]:
            raise ValueError(
                "category must be one of: AD, MCI, CN, All"
            )

        rows = PredictionPipeline.load_top1_cache(cache_path)

        if not rows:
            return {
                "best_samples": [],
                "errors": [
                    {
                        "error": (
                            f"No cache found or cache is empty: {cache_path}. "
                            "Build the top-1 cache first."
                        )
                    }
                ],
            }

        filtered = []

        for row in rows:
            if category != "All" and row["true_label"] != category:
                continue

            if int(row["is_volume_correct"]) != 1:
                continue

            if int(row["is_top1_correct"]) != 1:
                continue

            if float(row["top1_slice_confidence"]) < float(min_confidence):
                continue

            filtered.append(row)

        filtered_by_class = {
            "AD": [],
            "MCI": [],
            "CN": [],
        }

        for row in filtered:
            filtered_by_class[row["true_label"]].append(row)

        for cls in filtered_by_class:
            filtered_by_class[cls] = sorted(
                filtered_by_class[cls],
                key=lambda x: x["top1_slice_confidence"],
                reverse=True,
            )

        max_samples = int(max_samples)

        if category != "All":
            return {
                "best_samples": filtered_by_class[category][:max_samples],
                "errors": [],
            }

        classes = ["AD", "MCI", "CN"]

        base_per_class = max_samples // len(classes)
        remainder = max_samples % len(classes)

        class_quota = {}

        for i, cls in enumerate(classes):
            class_quota[cls] = base_per_class + (1 if i < remainder else 0)

        final_results = []
        used_paths = set()

        for cls in classes:
            selected = filtered_by_class[cls][: class_quota[cls]]

            for row in selected:
                final_results.append(row)
                used_paths.add(row["file_path"])

        if len(final_results) < max_samples:
            leftovers = []

            for cls in classes:
                for row in filtered_by_class[cls]:
                    if row["file_path"] not in used_paths:
                        leftovers.append(row)

            leftovers = sorted(
                leftovers,
                key=lambda x: x["top1_slice_confidence"],
                reverse=True,
            )

            remaining_slots = max_samples - len(final_results)

            for row in leftovers[:remaining_slots]:
                final_results.append(row)
                used_paths.add(row["file_path"])

        class_order = {
            "AD": 0,
            "MCI": 1,
            "CN": 2,
        }

        final_results = sorted(
            final_results,
            key=lambda x: (
                class_order.get(x["true_label"], 99),
                -x["top1_slice_confidence"],
            ),
        )

        return {
            "best_samples": final_results,
            "errors": [],
        }

    @staticmethod
    def create_prediction(file_path, true_label_name):
        """
        Run full XAI prediction on a selected .npy scan.

        This:
            1. Loads the scan.
            2. Runs the volume-level cascade prediction.
            3. Computes gradient-based slice importance.
            4. Selects the top-1 important slice automatically.
            5. Runs single-slice prediction on the top-1 slice.
            6. Computes Grad-CAM++.
            7. Computes SHAP.
            8. Saves the final composite XAI plot.
        """

        if true_label_name not in LABEL_MAP:
            raise ValueError(
                f"Invalid true_label_name: {true_label_name}. "
                f"Expected one of {list(LABEL_MAP.keys())}"
            )

        if not os.path.exists(file_path):
            raise FileNotFoundError(
                f"File does not exist:\n{file_path}"
            )

        loaded = ModelLoader.load_models()

        stage1 = loaded["stage1"]
        stage2 = loaded["stage2"]
        cascade_model = loaded["cascade_model"]

        true_label = LABEL_MAP[true_label_name]

        raw_volume = DataProcessor.load_volume(file_path)
        norm_volume = DataProcessor.apply_transforms(raw_volume)

        volume_input = norm_volume.unsqueeze(0).to(DEVICE)

        volume_pred_idx, volume_pred_name, volume_probs = (
            PredictionPipeline.cascade_predict_volume(
                stage1=stage1,
                stage2=stage2,
                volume_tensor=volume_input,
            )
        )

        slice_importance, target_slice_idx = XAIMethods.compute_slice_importance(
            cascade_model=cascade_model,
            raw_volume=raw_volume,
            target_class=int(true_label),
        )

        target_slice_raw = raw_volume[target_slice_idx]
        target_slice_norm = DataProcessor.imagenet_normalize(target_slice_raw)
        target_slice_input = target_slice_norm.unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            out = cascade_model(target_slice_input)
            probs = out[0].detach().cpu().numpy()

        pred_idx = int(np.argmax(probs))
        pred_name = CLASS_NAMES[pred_idx]

        display_img = target_slice_raw.mean(dim=0).detach().cpu().numpy()
        display_img = np.clip(display_img, 0, 1)

        gradcam_maps, target_layer_name = XAIMethods.compute_gradcam_maps(
            cascade_model=cascade_model,
            target_slice_input=target_slice_input,
        )

        shap_values = XAIMethods.compute_shap_values(
            cascade_model=cascade_model,
            norm_volume=norm_volume,
            target_slice_input=target_slice_input,
            target_slice_idx=target_slice_idx,
        )

        save_path = XAIPlotter.create_xai_plot(
            file_path=file_path,
            true_label_name=true_label_name,
            pred_name=pred_name,
            probs=probs,
            target_slice_idx=target_slice_idx,
            display_img=display_img,
            slice_importance=slice_importance,
            gradcam_maps=gradcam_maps,
            shap_values=shap_values,
        )

        return {
            "scan_name": os.path.basename(file_path),
            "file_path": file_path,
            "true_label": true_label_name,
            "volume_prediction": volume_pred_name,
            "volume_probs": volume_probs,
            "top1_slice": target_slice_idx,
            "slice_prediction": pred_name,
            "slice_probs": probs,
            "confidence": float(probs.max()),
            "stage1_checkpoint": loaded["stage1_checkpoint"],
            "stage2_checkpoint": loaded["stage2_checkpoint"],
            "target_layer": target_layer_name,
            "save_path": save_path,
        }