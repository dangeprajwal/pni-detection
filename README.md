# 🔬 AI-Based PNI Detection in Histopathology

Automated detection of **perineural invasion (PNI)** in H&E-stained histopathology images using deep learning.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://pni-detection.streamlit.app)

---

## What It Does

Upload a histopathology image and the AI will:

1. **Detect nerve structures** in the tissue (Stage 1)
2. **Classify perineural invasion** — whether tumour cells are invading detected nerves (Stage 2)

Results are displayed as an annotated image with bounding boxes:
- **Green boxes** = Nerve detected, PNI negative
- **Red boxes** = Nerve detected, PNI positive

---

## Performance (5-Fold Cross-Validation)

| Task | AUC-ROC |
|------|:-------:|
| Nerve Detection | **0.999** |
| PNI Classification | **0.979** |

Evaluated using patient-level stratified group cross-validation to prevent data leakage.

---

## Model Architecture

### Two-Stage Pipeline

```
H&E Image → Multi-Scale Patch Extraction → Phikon-v2 (Frozen) → Feature Vectors
                                                                      ↓
                                              Stage 1: Nerve Detection (LogReg)
                                                                      ↓
                                           Stage 2: PNI Classification (LogReg)
```

### Foundation Model: Phikon-v2

- **Architecture:** Vision Transformer Large (ViT-L), 303M parameters
- **Pre-training:** DINOv2 self-supervised learning on **460 million** pathology image tiles from 100+ public cohorts
- **Output:** 1,024-dimensional CLS token feature vector per patch
- **Usage:** Frozen feature extractor (no fine-tuning required)
- **Source:** [owkin/phikon-v2](https://huggingface.co/owkin/phikon-v2)

### Classifiers

- **Type:** Logistic Regression with StandardScaler (scikit-learn Pipeline)
- **Nerve Detector:** Trained on 1,412 patches (862 nerve + 550 non-nerve)
- **PNI Classifier:** Trained on 862 nerve patches (430 PNI+ / 432 PNI-)
- **Why Logistic Regression?** Simplest model, highest AUC, most interpretable, trains in seconds

### Inference Pipeline

- **Multi-scale extraction:** 512px (stride 256) + 384px (stride 192)
- **Spatial clustering:** Nearby detections merged within 350px radius
- **Thresholds:** Nerve detection 0.7, PNI classification 0.5 (adjustable)

---

## Dataset

- **982 H&E images** at 20x magnification (1920 x 954 px, JPEG)
- **Cancer type:** Oral cavity squamous cell carcinoma
- **241** PNI-positive nerve images, **212** PNI-negative nerve images
- **38** cancer-only images (no nerves), **48** images at other magnifications (2x, 5x, 10x)
- Nerve annotations extracted automatically from hand-drawn circles on paired annotated/clean images

---

## Cross-Magnification Generalisation

The model was trained **only on 20x images** but generalises across magnifications without retraining:

| Magnification | Images | Nerve Detection Rate |
|:---:|:---:|:---:|
| 2x | 5 | 100% |
| 5x | 11 | 100% |
| 10x | 32 | 100% |
| 20x (training) | 453 | 99.9% (AUC) |

---

## Quick Start

### Use the Web App

Visit the deployed app (no installation needed):

**[pni-detection.streamlit.app](https://pni-detection.streamlit.app)**

### Run Locally

```bash
git clone https://github.com/dangeprajwal/pni-detection.git
cd pni-detection
pip install -r requirements.txt
streamlit run streamlit_app.py
```

The app will open at `http://localhost:8501`. On first run, it downloads the Phikon-v2 model (~1.2 GB).

---

## File Structure

```
pni-detection/
├── streamlit_app.py       # Streamlit web interface
├── inference.py           # Two-stage inference pipeline
├── train_classifiers.py   # One-time classifier training script
├── requirements.txt       # Python dependencies
├── features.npz           # Pre-extracted Phikon-v2 features (1412 x 1024)
├── classifiers/
│   ├── nerve_clf.pkl      # Pre-trained nerve detection classifier
│   └── pni_clf.pkl        # Pre-trained PNI classification classifier
└── examples/
    ├── nerve_pni_positive.jpg
    ├── nerve_pni_negative.jpg
    └── no_nerve_example.jpg
```

---

## Technical Stack

| Component | Detail |
|-----------|--------|
| Foundation Model | Phikon-v2 (owkin/phikon-v2) |
| Classifiers | Logistic Regression (scikit-learn) |
| Deep Learning | PyTorch (CPU inference supported) |
| Feature Extraction | HuggingFace Transformers |
| Image Processing | OpenCV (headless) |
| Web Framework | Streamlit |
| Deployment | Streamlit Community Cloud |

---

## Comparison with Published Literature

| Study / Method | Cancer Type | Task | AUC |
|----------------|------------|------|:---:|
| **This Study** | **Oral SCC** | **Nerve Detection** | **0.999** |
| **This Study** | **Oral SCC** | **PNI Classification** | **0.979** |
| Pancreatic PNI (AI-enhanced) | Pancreatic | PNI Detection | 0.81–0.85 |
| Rectal Cancer Radiomics | Rectal | PNI Prediction | 0.88–0.91 |
| Colorectal (Systematic Review) | Colorectal | PNI Detection | 0.80–0.91 |
| Oral SCC (Domain-KEY) | Oral SCC | PNI Classification | 0.89 acc |

---

## Limitations

- Single cancer type (oral cavity SCC) — generalisability to other cancers not yet validated
- Single institution — staining and scanner variability not tested
- Field images (1920 x 954 px), not whole slide images (WSI)
- Patient grouping approximated via SSIM (clinical IDs preferred)
- Test set of 15 images — larger prospective validation needed

---

## Disclaimer

**Research Use Only.** This tool is intended for research and educational purposes. It has not been validated for clinical diagnostic use and should not replace professional pathological assessment.

---

## License

This project uses the [Phikon-v2 model](https://huggingface.co/owkin/phikon-v2) which is subject to its own license terms. Please refer to the model card for details.
