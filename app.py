import os
import tempfile

import streamlit as st

from mri_xai.config import DEVICE
from mri_xai.manifest_manager import ManifestManager


# ============================================================
# Streamlit page setup
# ============================================================

st.set_page_config(
    page_title="MRI Cascade XAI Prediction",
    page_icon="🧠",
    layout="wide",
)

st.title("Prediction")

st.write(
    "Select a `.npy` scan from an existing dataset folder or upload one manually. "
    "The app runs the two-stage cascade model, finds the top-1 important slice, "
    "and generates Grad-CAM++ and SHAP visualizations."
)


# ============================================================
# Sidebar settings
# ============================================================

st.sidebar.header("Input Settings")
st.sidebar.write(f"Device: `{DEVICE}`")

validation_root = st.sidebar.selectbox(
    "Validation folder",
    ["OASIS1", "OASIS2", "validation_stage1"],
    index=0,
)

manifest_path = f"{validation_root}_manifest.md"
top1_cache_path = f"{validation_root}_top1_cache.csv"

manifest_manager = ManifestManager(
    validation_root=validation_root,
    manifest_path=manifest_path,
)

# Build/update the lightweight manifest only.
# This does NOT run the model.
if os.path.exists(validation_root):
    manifest_manager.create_manifest()

# ============================================================
# Session state defaults
# ============================================================

default_state = {
    "selected_file_path": None,
    "true_label_name": None,
    "uploaded_original_name": None,
    "matched_rows": [],
    "best_samples": [],
    "scan_display_mode": "all_scans",
    "last_validation_root": validation_root,
    "last_category": "All",
    "last_top1_confidence": 0.80,
    "last_max_best_samples": 10,
}

for key, value in default_state.items():
    if key not in st.session_state:
        st.session_state[key] = value


# Reset when validation folder changes
if st.session_state["last_validation_root"] != validation_root:
    st.session_state["last_validation_root"] = validation_root
    st.session_state["scan_display_mode"] = "all_scans"
    st.session_state["best_samples"] = []
    st.session_state["selected_file_path"] = None
    st.session_state["true_label_name"] = None
    st.session_state["uploaded_original_name"] = None
    st.session_state["matched_rows"] = []


# ============================================================
# Best sample finder controls
# ============================================================

st.sidebar.subheader("Best Sample Finder")

category = st.sidebar.radio(
    "Category",
    ["All", "AD", "MCI", "CN"],
    index=0,
)

top1_confidence_threshold = st.sidebar.slider(
    "Minimum prediction confidence",
    min_value=0.50,
    max_value=0.99,
    value=0.80,
    step=0.01,
)

max_best_samples = st.sidebar.number_input(
    "Max best samples",
    min_value=1,
    max_value=50,
    value=10,
    step=1,
)

find_best_button = st.sidebar.button("Find best samples")
show_all_button = st.sidebar.button("Show all scans from dataset")


# Reset stale best-sample results when filter settings change
settings_changed = (
    st.session_state["last_category"] != category
    or float(st.session_state["last_top1_confidence"]) != float(top1_confidence_threshold)
    or int(st.session_state["last_max_best_samples"]) != int(max_best_samples)
)

if settings_changed:
    st.session_state["last_category"] = category
    st.session_state["last_top1_confidence"] = float(top1_confidence_threshold)
    st.session_state["last_max_best_samples"] = int(max_best_samples)

    st.session_state["best_samples"] = []
    st.session_state["scan_display_mode"] = "all_scans"
    st.session_state["selected_file_path"] = None
    st.session_state["true_label_name"] = None
    st.session_state["uploaded_original_name"] = None
    st.session_state["matched_rows"] = []


# ============================================================
# Input mode
# ============================================================

input_mode = st.sidebar.radio(
    "Choose input mode",
    [
        "Select from existing folder",
        "Upload .npy file",
    ],
)


# ============================================================
# Helper: get all scans from manifest
# ============================================================

def get_all_scans_from_manifest():
    if os.path.exists(validation_root):
        manifest_manager.create_manifest()

    rows = manifest_manager.load_manifest()

    all_scans = []

    for row in rows:
        file_path = os.path.join(validation_root, row["relative_path"])

        all_scans.append(
            {
                "file_name": row["file_name"],
                "file_path": file_path,
                "true_label": row["true_label"],
                "relative_path": row["relative_path"],
            }
        )

    return all_scans


# ============================================================
# Show all scans
# ============================================================

if show_all_button:
    if not os.path.exists(validation_root):
        st.error(f"Validation folder not found: `{validation_root}`")

    else:
        manifest_manager.create_manifest()

        st.session_state["scan_display_mode"] = "all_scans"
        st.session_state["best_samples"] = []
        st.session_state["selected_file_path"] = None
        st.session_state["true_label_name"] = None
        st.session_state["uploaded_original_name"] = None
        st.session_state["matched_rows"] = []

        st.success(f"Dropdown is now showing all scans from `{validation_root}`.")


# ============================================================
# Find best samples from CSV only
# ============================================================

if find_best_button:
    if not os.path.exists(top1_cache_path):
        st.error(
            "Best-sample data has not been prepared yet. "
            "Please run the precompute script first, then try again."
        )

    else:
        try:
            from mri_xai.prediction_pipeline import PredictionPipeline

            search_result = PredictionPipeline.find_best_samples_from_cache(
                cache_path=top1_cache_path,
                min_confidence=float(top1_confidence_threshold),
                max_samples=int(max_best_samples),
                category=category,
            )

            best_samples = search_result["best_samples"]
            errors = search_result["errors"]

            st.session_state["best_samples"] = best_samples
            st.session_state["scan_display_mode"] = "best_samples"
            st.session_state["selected_file_path"] = None
            st.session_state["true_label_name"] = None
            st.session_state["uploaded_original_name"] = None
            st.session_state["matched_rows"] = []

            if not best_samples:
                st.warning(
                    f"No best samples found in `{top1_cache_path}` for category "
                    f"`{category}` with top-1 confidence >= "
                    f"{top1_confidence_threshold:.2f}."
                )

            else:
                st.success(
                    f"Found {len(best_samples)} best samples from `{top1_cache_path}` "
                    f"for category `{category}` using top-1 confidence >= "
                    f"{top1_confidence_threshold:.2f}."
                )

            if errors:
                with st.expander(f"Cache search messages ({len(errors)})"):
                    st.dataframe(errors, use_container_width=True)

        except Exception as e:
            st.error("Best-sample CSV search failed.")
            st.exception(e)


# ============================================================
# Select from existing folder
# ============================================================

if input_mode == "Select from existing folder":

    if not os.path.exists(validation_root):
        st.warning(
            f"Folder not found: `{validation_root}`. "
            "Make sure this folder exists in your project root."
        )

    else:
        scan_display_mode = st.session_state["scan_display_mode"]

        if scan_display_mode == "best_samples":
            best_samples = st.session_state["best_samples"]

            if not best_samples:
                st.warning(
                    "No best samples are currently available. "
                    "Click `Find best samples from CSV` or `Show all scans from dataset`."
                )

            else:
                display_names = [
                    f"{row['true_label']} | "
                    f"important-slice-conf={float(row['top1_slice_confidence']):.3f} | "
                    f"slice={int(row['top1_slice_index'])} | "
                    f"{row['file_name']}"
                    for row in best_samples
                ]

                selected_display_name = st.selectbox(
                    f"Select scan from {validation_root}",
                    display_names,
                    key=(
                        f"best_sample_select_"
                        f"{validation_root}_"
                        f"{category}_"
                        f"{top1_confidence_threshold}_"
                        f"{max_best_samples}"
                    ),
                )

                selected_index = display_names.index(selected_display_name)
                selected_row = best_samples[selected_index]

                selected_file_path = selected_row["file_path"]
                true_label_name = selected_row["true_label"]

                st.session_state["selected_file_path"] = selected_file_path
                st.session_state["true_label_name"] = true_label_name
                st.session_state["uploaded_original_name"] = None
                st.session_state["matched_rows"] = []

        else:
            all_scans = get_all_scans_from_manifest()

            if not all_scans:
                st.warning(f"No `.npy` files found inside `{validation_root}`.")

            else:
                display_names = [
                    f"{row['true_label']} | {row['relative_path']}"
                    for row in all_scans
                ]

                selected_display_name = st.selectbox(
                    f"Select scan from {validation_root}",
                    display_names,
                    key=f"all_scan_select_{validation_root}",
                )

                selected_index = display_names.index(selected_display_name)
                selected_row = all_scans[selected_index]

                selected_file_path = selected_row["file_path"]
                true_label_name = selected_row["true_label"]

                st.session_state["selected_file_path"] = selected_file_path
                st.session_state["true_label_name"] = true_label_name
                st.session_state["uploaded_original_name"] = None
                st.session_state["matched_rows"] = []


# ============================================================
# Manual upload
# ============================================================

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

        st.session_state["selected_file_path"] = selected_file_path
        st.session_state["true_label_name"] = true_label_name
        st.session_state["uploaded_original_name"] = uploaded_original_name
        st.session_state["matched_rows"] = matched_rows


# ============================================================
# Get current selected sample
# ============================================================

selected_file_path = st.session_state["selected_file_path"]
true_label_name = st.session_state["true_label_name"]
uploaded_original_name = st.session_state["uploaded_original_name"]
matched_rows = st.session_state["matched_rows"]


# ============================================================
# Create prediction
# ============================================================

create_button = st.button("Create Prediction", type="primary")

if create_button:

    if selected_file_path is None:
        st.error("Please select, upload, or find a best `.npy` scan first.")

    elif true_label_name is None:
        st.error("Could not determine the true label.")

    else:
        try:
            with st.spinner("Running prediction and generating XAI plot..."):
                from mri_xai.prediction_pipeline import PredictionPipeline

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
                st.write(f"Top-1 cache used: `{top1_cache_path}`")
                st.write(f"Scan display mode: `{st.session_state['scan_display_mode']}`")

                if uploaded_original_name is not None:
                    st.write(f"Uploaded original filename: `{uploaded_original_name}`")

                if matched_rows:
                    st.write("Manifest matches:")
                    st.write(matched_rows)

        except Exception as e:
            st.error("Prediction failed.")
            st.exception(e)