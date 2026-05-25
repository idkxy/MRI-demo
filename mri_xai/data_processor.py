import os
import numpy as np
import torch

from mri_xai.config import (
    TARGET_SLICES,
    IMAGENET_MEAN,
    IMAGENET_STD,
)


class DataProcessor:
    @staticmethod
    def imagenet_normalize(x):
        # x: (3, H, W), expected in [0, 1]
        return (x - IMAGENET_MEAN.to(x.device)) / IMAGENET_STD.to(x.device)

    @staticmethod
    def load_volume(fpath):
        data = np.load(fpath)
        data = torch.from_numpy(data).float()

        # Expected final shape: (S, 3, H, W)
        if data.dim() == 3:
            # (S, H, W)
            data = data.unsqueeze(1).repeat(1, 3, 1, 1)

        elif data.dim() == 4:
            if data.shape[1] == 1:
                # (S, 1, H, W)
                data = data.repeat(1, 3, 1, 1)

            elif data.shape[1] == 3:
                # already (S, 3, H, W)
                pass

            elif data.shape[-1] in [1, 3]:
                # (S, H, W, C)
                data = data.permute(0, 3, 1, 2)

                if data.shape[1] == 1:
                    data = data.repeat(1, 3, 1, 1)

            else:
                raise ValueError(f"Unexpected 4D shape: {tuple(data.shape)}")

        else:
            raise ValueError(f"Unexpected data shape: {tuple(data.shape)}")

        if data.max() > 1.5:
            data = data / 255.0

        data = torch.clamp(data, 0, 1)

        # Force 30 slices.
        s = data.shape[0]

        if s < TARGET_SLICES:
            pad = data[-1:].repeat(TARGET_SLICES - s, 1, 1, 1)
            data = torch.cat([data, pad], dim=0)

        elif s > TARGET_SLICES:
            start = (s - TARGET_SLICES) // 2
            data = data[start:start + TARGET_SLICES]

        return data

    @staticmethod
    def apply_transforms(volume):
        transformed = []

        for i in range(volume.shape[0]):
            transformed.append(DataProcessor.imagenet_normalize(volume[i]))

        return torch.stack(transformed)

    @staticmethod
    def infer_label_from_path(file_path):
        parts = os.path.normpath(file_path).split(os.sep)

        for label in ["AD", "MCI", "CN"]:
            if label in parts:
                return label

        for label in ["ad", "mci", "cn"]:
            if label in parts:
                return label.upper()

        return None