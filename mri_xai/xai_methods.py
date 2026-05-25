import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import shap

from mri_xai.config import DEVICE, CLASS_NAMES
from mri_xai.data_processor import DataProcessor


class GradCAMPlusPlus:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer

        self.activations = None
        self.gradients = None

        self._fwd = self.target_layer.register_forward_hook(self._forward_hook)
        self._bwd = self.target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, inp, output):
        self.activations = output

    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def __call__(self, x, class_idx):
        self.model.zero_grad(set_to_none=True)

        output = self.model(x)
        score = output[:, class_idx].sum()

        score.backward(retain_graph=True)

        if self.activations is None or self.gradients is None:
            raise RuntimeError("No activations or gradients captured.")

        A = self.activations
        G = self.gradients

        G2 = G ** 2
        G3 = G ** 3

        sum_A = A.sum(dim=(2, 3), keepdim=True)

        eps = 1e-8

        alpha_num = G2
        alpha_den = 2.0 * G2 + sum_A * G3

        alpha_den = torch.where(
            alpha_den != 0,
            alpha_den,
            torch.ones_like(alpha_den) * eps,
        )

        alphas = alpha_num / (alpha_den + eps)
        weights = torch.sum(alphas * F.relu(G), dim=(2, 3), keepdim=True)

        cam = torch.sum(weights * A, dim=1, keepdim=True)
        cam = F.relu(cam)

        cam = F.interpolate(
            cam,
            size=x.shape[2:],
            mode="bilinear",
            align_corners=False,
        )

        cam = cam[0, 0]

        cmin, cmax = cam.min(), cam.max()

        if cmax - cmin > 1e-8:
            cam = (cam - cmin) / (cmax - cmin)
        else:
            cam = torch.zeros_like(cam)

        return cam.detach().cpu().numpy()

    def remove_hooks(self):
        self._fwd.remove()
        self._bwd.remove()


class XAIMethods:
    @staticmethod
    def find_last_conv_layer(model):
        last_name = None
        last_module = None

        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d):
                last_name = name
                last_module = module

        if last_module is None:
            raise RuntimeError("No Conv2d layer found.")

        return last_name, last_module

    @staticmethod
    def compute_slice_importance(cascade_model, raw_volume, target_class):
        slice_grad_norms = []

        cascade_model.eval()

        for i in range(raw_volume.shape[0]):
            sl = raw_volume[i].clone().detach().to(DEVICE)

            sl_norm = DataProcessor.imagenet_normalize(sl).unsqueeze(0)
            sl_norm = sl_norm.clone().detach().requires_grad_(True)
            sl_norm.retain_grad()

            out = cascade_model(sl_norm)
            score = out[0, target_class]

            cascade_model.zero_grad(set_to_none=True)

            if sl_norm.grad is not None:
                sl_norm.grad.zero_()

            score.backward()

            grad = sl_norm.grad.detach().cpu()
            grad_norm = grad.norm().item()

            slice_grad_norms.append(grad_norm)

        slice_importance = np.array(slice_grad_norms, dtype=np.float32)

        if slice_importance.max() > 0:
            slice_importance = slice_importance / slice_importance.max()

        top1_idx = int(np.argmax(slice_importance))

        return slice_importance, top1_idx

    @staticmethod
    def compute_gradcam_maps(cascade_model, target_slice_input):
        target_layer_name, target_layer = XAIMethods.find_last_conv_layer(cascade_model)

        gradcam_pp = GradCAMPlusPlus(cascade_model, target_layer)

        gradcam_maps = {}

        for class_idx, cname in enumerate(CLASS_NAMES):
            x_in = target_slice_input.clone().detach().requires_grad_(True)
            cam_map = gradcam_pp(x_in, class_idx)
            gradcam_maps[cname] = cam_map

        gradcam_pp.remove_hooks()

        return gradcam_maps, target_layer_name

    @staticmethod
    def compute_shap_values(cascade_model, norm_volume, target_slice_input, target_slice_idx):
        background_slices = []

        for i in range(norm_volume.shape[0]):
            if i == target_slice_idx:
                continue

            background_slices.append(norm_volume[i])

        background = torch.stack(background_slices).float().to(DEVICE)

        explainer = shap.GradientExplainer(cascade_model, background)
        shap_values = explainer.shap_values(target_slice_input)
        shap_values = np.array(shap_values)

        return shap_values

    @staticmethod
    def get_class_shap(shap_values, class_idx):
        if shap_values.ndim == 5 and shap_values.shape[-1] == len(CLASS_NAMES):
            s = shap_values[0, :, :, :, class_idx]
            return s.mean(axis=0)

        elif shap_values.ndim == 5 and shap_values.shape[0] == len(CLASS_NAMES):
            s = shap_values[class_idx, 0]
            return s.mean(axis=0)

        else:
            raise ValueError(f"Unexpected SHAP shape: {shap_values.shape}")