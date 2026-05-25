import os
import glob
import tempfile

import streamlit as st

from mri_xai.config import DEVICE
from mri_xai.data_processor import DataProcessor
from mri_xai.manifest_manager import ManifestManager
from mri_xai.prediction_pipeline import PredictionPipeline


st.set_page_config(
    page_title="MRI Cascade XAI Prediction",
    page_icon="🧠",
    layout="wide",
)

st.title("MRI Alzheimer’s Cascade XAI Prediction")

st.write(
    "Select a `.npy` scan from `validation_stage1` or upload one manually. "
    "When **Create Prediction** is clicked, the app runs the cascade model, "
    "finds the top-1 important slice, and generates Grad-CAM++ and SHAP plots."
)


st.sidebar.header("Input Settings")
st.sidebar.write(f"Device: `{DEVICE}`")

validation_root = st.sidebar.text_input(
    "Validation folder path",
    value="validation_stage1",
)

manifest_path = st.sidebar.text_input(
    "Manifest path",
    value="validation_manifest.md",
)

manifest_manager = ManifestManager(
    validation_root=validation_root,
    manifest_path=manifest_path,
)


st.sidebar.subheader("Validation Manifest")

if st.sidebar.button("Rebuild validation manifest"):
    if os.path.exists(validation_root):
        rows = manifest_manager.create_manifest()
        st.sidebar.success(f"Manifest rebuilt with {len(rows)} files.")
        st.sidebar.write(f"Saved to: `{manifest_path}`")
    else:
        st.sidebar.error(f"Validation folder not found: `{validation_root}`")


if os.path.exists(validation_root) and not os.path.exists(manifest_path):
    manifest_manager.create_manifest()


input_mode = st.sidebar.radio(
    "Choose input mode",
    [
        "Select from validation_stage1 folder",
        "Upload .npy file",
    ],
)

selected_file_path = None
true_label_name = None
uploaded_original_name = None
matched_rows = []


if input_mode == "Select from validation_stage1 folder":

    npy_files = []

    if os.path.exists(validation_root):
        npy_files = sorted(
            glob.glob(
                os.path.join(validation_root, "**", "*.npy"),
                recursive=True,
            )
        )

    if not os.path.exists(validation_root):
        st.warning(
            f"Folder not found: `{validation_root}`. "
            "Make sure you run Streamlit from your project root."
        )

    elif not npy_files:
        st.warning(f"No `.npy` files found inside `{validation_root}`.")

    else:
        display_names = [
            os.path.relpath(path, validation_root)
            for path in npy_files
        ]

        selected_display_name = st.selectbox(
            "Select scan from validation_stage1",
            display_names,
        )

        selected_index = display_names.index(selected_display_name)
        selected_file_path = npy_files[selected_index]

        inferred_label = DataProcessor.infer_label_from_path(selected_file_path)

        if inferred_label is not None:
            true_label_name = inferred_label

            st.success(
                f"Selected file: `{selected_file_path}`\n\n"
                f"True label inferred from folder: `{true_label_name}`"
            )

        else:
            st.warning(
                "Could not infer true label from folder path. "
                "Please select the true label manually."
            )

            true_label_name = st.sidebar.selectbox(
                "True label",
                ["AD", "MCI", "CN"],
                index=0,
            )


else:
    uploaded_file = st.file_uploader(
        "Upload scan slices `.npy` file",
        type=["npy"],
    )

    if uploaded_file is not None:
        uploaded_original_name = uploaded_file.name

        tmp_dir = tempfile.mkdtemp()
        selected_file_path = os.path.join(tmp_dir, uploaded_file.name)

        with open(selected_file_path, "wb") as f:
            f.write(uploaded_file.read())

        st.info(f"Uploaded file: `{uploaded_file.name}`")

        if os.path.exists(validation_root):
            manifest_manager.create_manifest()

        inferred_label, matched_rows = manifest_manager.find_label_by_filename(
            uploaded_file.name
        )

        if inferred_label is not None:
            true_label_name = inferred_label

            st.success(
                f"Uploaded filename matched one file in `{manifest_path}`.\n\n"
                f"True label inferred as: `{true_label_name}`"
            )

            with st.expander("Matched validation record"):
                st.write(matched_rows[0])

        elif len(matched_rows) > 1:
            st.warning(
                "Multiple files with the same name were found in the manifest. "
                "Please select the correct validation record."
            )

            options = [
                f"{row['true_label']} | {row['relative_path']}"
                for row in matched_rows
            ]

            selected_match = st.selectbox(
                "Select matching validation record",
                options,
            )

            selected_idx = options.index(selected_match)
            true_label_name = matched_rows[selected_idx]["true_label"]

            st.info(f"Selected true label: `{true_label_name}`")

        else:
            st.warning(
                f"No filename match found for `{uploaded_file.name}` "
                f"in `{manifest_path}`.\n\n"
                "Please select the true label manually."
            )

            true_label_name = st.sidebar.selectbox(
                "True label",
                ["AD", "MCI", "CN"],
                index=0,
            )


st.divider()

create_button = st.button("Create Prediction", type="primary")

if create_button:

    if selected_file_path is None:
        st.error("Please select or upload a `.npy` scan first.")

    elif true_label_name is None:
        st.error("Could not determine the true label.")

    else:
        try:
            with st.spinner("Running prediction and generating XAI plot..."):
                result = PredictionPipeline.create_prediction(
                    file_path=selected_file_path,
                    true_label_name=true_label_name,
                )

            st.success("Prediction created successfully.")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Volume Prediction", result["volume_prediction"])

            with col2:
                st.metric("Top-1 Slice", result["top1_slice"])

            with col3:
                st.metric("Slice Prediction", result["slice_prediction"])

            st.subheader("Slice Prediction Probabilities")

            prob_col1, prob_col2, prob_col3 = st.columns(3)

            with prob_col1:
                st.metric("CN", f"{result['slice_probs'][0]:.4f}")

            with prob_col2:
                st.metric("MCI", f"{result['slice_probs'][1]:.4f}")

            with prob_col3:
                st.metric("AD", f"{result['slice_probs'][2]:.4f}")

            st.subheader("Generated XAI Plot")

            st.image(result["save_path"], use_container_width=True)

            with open(result["save_path"], "rb") as f:
                st.download_button(
                    label="Download XAI Plot",
                    data=f.read(),
                    file_name=os.path.basename(result["save_path"]),
                    mime="image/png",
                )

            with st.expander("Details"):
                st.write(f"Scan: `{result['scan_name']}`")
                st.write(f"File path: `{result['file_path']}`")
                st.write(f"True label: `{result['true_label']}`")
                st.write(f"Volume prediction: `{result['volume_prediction']}`")
                st.write(f"Slice prediction: `{result['slice_prediction']}`")
                st.write(f"Confidence: `{result['confidence']:.4f}`")
                st.write(f"Stage 1 checkpoint: `{result['stage1_checkpoint']}`")
                st.write(f"Stage 2 checkpoint: `{result['stage2_checkpoint']}`")
                st.write(f"Grad-CAM++ target layer: `{result['target_layer']}`")
                st.write(f"Saved to: `{result['save_path']}`")
                st.write(f"Manifest used: `{manifest_path}`")

                if uploaded_original_name is not None:
                    st.write(f"Uploaded original filename: `{uploaded_original_name}`")

                if matched_rows:
                    st.write("Manifest matches:")
                    st.write(matched_rows)

        except Exception as e:
            st.error("Prediction failed.")
            st.exception(e)