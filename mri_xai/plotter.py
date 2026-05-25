import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from mri_xai.config import CLASS_NAMES
from mri_xai.xai_methods import XAIMethods


class XAIPlotter:
    @staticmethod
    def create_xai_plot(
        file_path,
        true_label_name,
        pred_name,
        probs,
        target_slice_idx,
        display_img,
        slice_importance,
        gradcam_maps,
        shap_values,
        save_dir="results/xai",
    ):
        fig = plt.figure(figsize=(16, 11))

        gs = gridspec.GridSpec(
            3,
            3,
            figure=fig,
            width_ratios=[1.1, 1.1, 1.4],
            height_ratios=[1, 1, 1],
            wspace=0.3,
            hspace=0.35,
        )

        ax_orig = fig.add_subplot(gs[0, 0])
        ax_orig.imshow(display_img, cmap="gray", vmin=0, vmax=1)
        ax_orig.set_title(f"Original Top-1 Slice {target_slice_idx}", fontsize=11)
        ax_orig.axis("off")

        ax_bar = fig.add_subplot(gs[1:, 0])

        bar_colors = [
            "tab:red" if i == target_slice_idx else "tab:blue"
            for i in range(len(slice_importance))
        ]

        ax_bar.bar(range(len(slice_importance)), slice_importance, color=bar_colors)
        ax_bar.set_xlabel("Slice Index")
        ax_bar.set_ylabel("Gradient Attribution (L2 norm, normalized)")
        ax_bar.set_title("Slice Importance", fontsize=11)
        ax_bar.set_ylim(0, 1.05)

        global_shap_abs = []

        for i in range(3):
            global_shap_abs.append(np.abs(XAIMethods.get_class_shap(shap_values, i)))

        global_shap_max = np.percentile(
            np.concatenate([s.ravel() for s in global_shap_abs]),
            99,
        )

        if global_shap_max < 1e-10:
            global_shap_max = 1e-3

        for row, (cname, prob) in enumerate(zip(CLASS_NAMES, probs)):

            ax_cam = fig.add_subplot(gs[row, 1])

            cam_map = gradcam_maps[cname]

            ax_cam.imshow(display_img, cmap="gray", vmin=0, vmax=1)
            ax_cam.imshow(
                cam_map,
                cmap="jet",
                alpha=0.4,
                interpolation="bilinear",
            )

            ax_cam.set_title(f"Grad-CAM++ → {cname} (p={prob:.3f})", fontsize=11)
            ax_cam.axis("off")

            ax_shap = fig.add_subplot(gs[row, 2])

            sv = XAIMethods.get_class_shap(shap_values, row)

            im = ax_shap.imshow(
                sv,
                cmap="RdBu_r",
                vmin=-global_shap_max,
                vmax=global_shap_max,
                interpolation="nearest",
            )

            ax_shap.set_title(f"SHAP → {cname}\nProb: {prob:.3f}", fontsize=11)
            ax_shap.axis("off")

            fig.colorbar(im, ax=ax_shap, fraction=0.046, pad=0.04)

        fig.suptitle(
            f"XAI | {os.path.basename(file_path)} | "
            f"True: {true_label_name} | "
            f"Pred: {pred_name} "
            f"(conf: {probs.max():.3f}) | "
            f"Top-1 Slice: {target_slice_idx}",
            fontsize=13,
            fontweight="bold",
            y=0.98,
        )

        os.makedirs(save_dir, exist_ok=True)

        clean_file_name = os.path.basename(file_path).replace(".npy", "")

        save_path = os.path.join(
            save_dir,
            f"xai_{true_label_name}_{clean_file_name}_top1_slice_{target_slice_idx}.png",
        )

        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

        return save_path