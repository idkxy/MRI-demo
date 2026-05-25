import os
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

                return pred_idx, CLASS_NAMES[pred_idx], probs

            s2_out = stage2(volume_tensor)
            s2_prob = torch.softmax(s2_out, dim=1)

            p_cn = s1_prob[0, 0].item()
            p_mci = s1_prob[0, 1].item() * s2_prob[0, 0].item()
            p_ad = s1_prob[0, 1].item() * s2_prob[0, 1].item()

            probs = np.array([p_cn, p_mci, p_ad])

            pred_idx = int(np.argmax(probs))

            return pred_idx, CLASS_NAMES[pred_idx], probs

    @staticmethod
    def create_prediction(file_path, true_label_name):
        if true_label_name not in LABEL_MAP:
            raise ValueError(
                f"Invalid true_label_name: {true_label_name}. "
                f"Expected one of {list(LABEL_MAP.keys())}"
            )

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File does not exist:\n{file_path}")

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