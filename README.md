# MRI Cascade XAI Demo

This project provides a Streamlit web interface for running a two-stage Alzheimer’s disease classification pipeline on preprocessed MRI `.npy` scan slices.

The app predicts the class label and generates explainability visualizations using:

- Slice-importance gradients
- Grad-CAM++
- SHAP

The app automatically selects the MRI slice with the highest gradient importance and creates a final XAI visualization.

---

## Project Structure

```text
MRI-demo/
├── app.py
├── mri_xai/
│   ├── __init__.py
│   ├── config.py
│   ├── models.py
│   ├── model_loader.py
│   ├── data_processor.py
│   ├── manifest_manager.py
│   ├── xai_methods.py
│   ├── plotter.py
│   └── prediction_pipeline.py
├── models/
│   ├── stage1_CN_vs_Impaired_30slices_WARMUP_score_....pth
│   └── stage2_MCI_vs_AD_30slices_WARMUP_score_....pth
├── validation_stage1/
│   ├── AD/
│   ├── MCI/
│   └── CN/
├── results/
├── requirements.txt
└── README.md
```

---

## Requirements

This project requires Python 3.10 or later.

Install the dependencies inside a virtual environment.

```bash
cd /path/to/MRI-demo

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If you do not have a `requirements.txt` yet, create one with the following content:

```text
streamlit
torch
torchvision
matplotlib
shap
numpy
```

Then install the dependencies:

```bash
python -m pip install -r requirements.txt
```

---

## Model Checkpoints

Place the trained Stage 1 and Stage 2 model checkpoints inside the `models/` folder.

Expected filename patterns:

```text
models/stage1_CN_vs_Impaired_30slices_WARMUP_score_*.pth
models/stage2_MCI_vs_AD_30slices_WARMUP_score_*.pth
```

The app automatically loads the latest matching checkpoint based on sorted filename order.

---

## Validation Data

Place the preprocessed `.npy` scan files inside `validation_stage1/` using this structure:

```text
validation_stage1/
├── AD/
│   └── sample_ad.npy
├── MCI/
│   └── sample_mci.npy
└── CN/
    └── sample_cn.npy
```

Each `.npy` file should contain a scan volume or scan slices that can be converted to the expected format:

```text
(S, 3, H, W)
```

where:

```text
S = number of slices
H = image height
W = image width
```

The app automatically forces each scan to 30 slices if needed.

---

## Running the App

From the project root, run:

```bash
cd /path/to/MRI-demo
python -m streamlit run app.py
```

Then open the local URL shown in the terminal.

Usually, it will be:

```text
http://localhost:8501
```

---

## Using the App

1. Open the Streamlit app in your browser.

2. Check that the validation folder path is correct. The default is:

```text
validation_stage1
```

3. The app automatically creates a manifest file:

```text
validation_manifest.md
```

This file maps each `.npy` filename to its true label based on the folder it is stored in.

For example:

```markdown
| file_name | true_label | relative_path |
|---|---|---|
| sample_ad.npy | AD | AD/sample_ad.npy |
| sample_mci.npy | MCI | MCI/sample_mci.npy |
| sample_cn.npy | CN | CN/sample_cn.npy |
```

4. Choose an input mode:

```text
Select from validation_stage1 folder
```

or:

```text
Upload .npy file
```

5. If selecting from `validation_stage1`, the true label is inferred from the folder name.

6. If uploading a `.npy` file manually, the app checks the uploaded filename against `validation_manifest.md` to infer the true label.

7. Click:

```text
Create Prediction
```

8. The app will:

- Load the selected scan
- Run the two-stage cascade classifier
- Calculate slice importance
- Automatically select the slice with the highest importance
- Generate Grad-CAM++ visualizations
- Generate SHAP visualizations
- Save the final XAI image under `results/xai/`

---

## Apple Silicon / MPS Support

On Apple Silicon Macs, the app can use the Apple GPU through PyTorch MPS if available.

To check MPS support, run:

```bash
python -c "import torch; print(torch.backends.mps.is_available()); print(torch.backends.mps.is_built())"
```

If both values are:

```text
True
True
```

then the app should show:

```text
Device: mps
```

in the Streamlit sidebar.

If MPS is not available, the app will use CPU.

---

## Output

Generated plots are saved to:

```text
results/xai/
```

Each output image includes:

- Original top-1 important MRI slice
- Slice-importance bar chart
- Grad-CAM++ maps for CN, MCI, and AD
- SHAP maps for CN, MCI, and AD
- Predicted class
- Confidence score

