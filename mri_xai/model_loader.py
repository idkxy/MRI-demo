import os
import glob
import torch
import streamlit as st

from mri_xai.config import DEVICE
from mri_xai.models import ADEfficientNet, SingleSliceWrapper, CascadeSliceModel


class ModelLoader:
    @staticmethod
    @st.cache_resource
    def load_models():
        stage1 = ADEfficientNet(num_classes=2, dropout=0.4).to(DEVICE)
        stage2 = ADEfficientNet(num_classes=2, dropout=0.4).to(DEVICE)

        s1_path = glob.glob("models/stage1_CN_vs_Impaired_30slices_WARMUP_score_0.7824_epoch_31.pth")
        s2_path = glob.glob("models/stage2_MCI_vs_AD_30slices_WARMUP_score_0.8879_epoch_54.pth")

        if not s1_path:
            raise FileNotFoundError("Stage 1 checkpoint not found in models/")

        if not s2_path:
            raise FileNotFoundError("Stage 2 checkpoint not found in models/")

        s1_ckpt = sorted(s1_path)[-1]
        s2_ckpt = sorted(s2_path)[-1]

        try:
            stage1.load_state_dict(
                torch.load(s1_ckpt, map_location=DEVICE, weights_only=True)
            )
            stage2.load_state_dict(
                torch.load(s2_ckpt, map_location=DEVICE, weights_only=True)
            )

        except TypeError:
            stage1.load_state_dict(torch.load(s1_ckpt, map_location=DEVICE))
            stage2.load_state_dict(torch.load(s2_ckpt, map_location=DEVICE))

        stage1.eval()
        stage2.eval()

        ss_stage1 = SingleSliceWrapper(stage1).to(DEVICE).eval()
        ss_stage2 = SingleSliceWrapper(stage2).to(DEVICE).eval()

        cascade_model = CascadeSliceModel(ss_stage1, ss_stage2).to(DEVICE).eval()

        return {
            "stage1": stage1,
            "stage2": stage2,
            "cascade_model": cascade_model,
            "stage1_checkpoint": os.path.basename(s1_ckpt),
            "stage2_checkpoint": os.path.basename(s2_ckpt),
        }