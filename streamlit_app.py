"""
PNI Detection in Histopathology — Streamlit Web App

AI-powered two-stage detection of perineural invasion in H&E-stained
histopathology images using the Phikon-v2 foundation model.

Stage 1: Nerve detection (identifies nerve structures)
Stage 2: PNI classification (determines if tumour invades the nerve)
"""

import streamlit as st
import torch
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
from transformers import AutoModel, AutoImageProcessor

from inference import run_inference

# ── Page config ───────────────────────────────────────────────────────

st.set_page_config(
    page_title="PNI Detection in Histopathology",
    page_icon="🔬",
    layout="wide",
)

# ── Cached model loading (runs once) ─────────────────────────────────

@st.cache_resource(show_spinner="Loading Phikon-v2 foundation model...")
def load_model():
    device = "cuda" if torch.cuda.is_available() else (
        "mps" if torch.backends.mps.is_available() else "cpu"
    )
    dtype = torch.float16 if device == "cuda" else torch.float32
    model = AutoModel.from_pretrained(
        "owkin/phikon-v2",
        trust_remote_code=True,
        torch_dtype=dtype,
    ).to(device).eval()
    processor = AutoImageProcessor.from_pretrained(
        "owkin/phikon-v2",
        trust_remote_code=True,
        use_fast=True,
    )
    return model, processor, device


@st.cache_resource(show_spinner="Loading classifiers...")
def load_classifiers():
    base = Path(__file__).parent
    nerve_clf = joblib.load(base / "classifiers" / "nerve_clf.pkl")
    pni_clf = joblib.load(base / "classifiers" / "pni_clf.pkl")
    return nerve_clf, pni_clf


# ── Load everything ──────────────────────────────────────────────────

MODEL, PROCESSOR, DEVICE = load_model()
NERVE_CLF, PNI_CLF = load_classifiers()

# ── Sidebar: Advanced Settings ───────────────────────────────────────

with st.sidebar:
    st.header("⚙️ Settings")

    with st.expander("Microscope UI Overlay Removal", expanded=False):
        st.caption("Set pixel values to crop if your microscope adds a scale bar or metadata overlay.")
        crop_top = st.number_input("Crop Top (px)", min_value=0, max_value=500, value=0)
        crop_bottom = st.number_input("Crop Bottom (px)", min_value=0, max_value=500, value=0)
        crop_left = st.number_input("Crop Left (px)", min_value=0, max_value=500, value=0)
        crop_right = st.number_input("Crop Right (px)", min_value=0, max_value=500, value=0)

    with st.expander("Detection Thresholds", expanded=False):
        nerve_threshold = st.slider(
            "Nerve Detection Threshold",
            min_value=0.50, max_value=0.95, value=0.70, step=0.05,
            help="Higher = fewer but more confident detections",
        )
        pni_threshold = st.slider(
            "PNI Classification Threshold",
            min_value=0.30, max_value=0.80, value=0.50, step=0.05,
            help="Higher = more specific, lower = more sensitive",
        )

    st.divider()
    st.caption("**Research Use Only.** Not validated for clinical diagnostic use.")

# ── Main UI ──────────────────────────────────────────────────────────

st.title("🔬 AI-Based PNI Detection in Histopathology")

st.markdown("""
Upload an H&E-stained histopathology image to detect **nerve structures**
and classify **perineural invasion (PNI)**.

The model uses [Phikon-v2](https://huggingface.co/owkin/phikon-v2), a
pathology foundation model trained on 460 million tiles,
with lightweight classifiers for nerve detection (AUC 0.999) and PNI
classification (AUC 0.979).

**Results:** 🟩 Green boxes = nerve without PNI · 🟥 Red boxes = nerve with PNI
""")

# ── Image Upload ─────────────────────────────────────────────────────

uploaded_file = st.file_uploader(
    "Upload H&E Image",
    type=["jpg", "jpeg", "png", "tif", "tiff", "bmp"],
    help="Drag and drop or click to upload a histopathology image",
)

# ── Example Images ───────────────────────────────────────────────────

example_dir = Path(__file__).parent / "examples"
if example_dir.exists() and not uploaded_file:
    st.markdown("**Or try an example image:**")
    example_files = sorted(example_dir.glob("*.jpg"))
    if example_files:
        cols = st.columns(len(example_files))
        for i, ex in enumerate(example_files):
            with cols[i]:
                img = Image.open(ex)
                st.image(img, caption=ex.stem.replace("_", " ").title(), use_container_width=True)
                if st.button(f"Use this", key=f"ex_{i}"):
                    st.session_state["example_image"] = str(ex)
                    st.rerun()

# ── Determine which image to analyze ─────────────────────────────────

image_array = None

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    image_array = np.array(image)
elif "example_image" in st.session_state:
    image = Image.open(st.session_state["example_image"]).convert("RGB")
    image_array = np.array(image)

# ── Run Inference ────────────────────────────────────────────────────

if image_array is not None:
    analyze = st.button("🔍 Analyze", type="primary", use_container_width=True)

    if analyze:
        with st.spinner("Analyzing image... This may take 10-30 seconds on first run."):
            annotated, verdict, regions = run_inference(
                image=image_array,
                model=MODEL,
                processor=PROCESSOR,
                nerve_clf=NERVE_CLF,
                pni_clf=PNI_CLF,
                device=DEVICE,
                crop_top=int(crop_top),
                crop_bottom=int(crop_bottom),
                crop_left=int(crop_left),
                crop_right=int(crop_right),
                nerve_threshold=nerve_threshold,
                pni_threshold=pni_threshold,
            )

        # ── Display Results ──────────────────────────────────────

        st.divider()

        # Verdict
        if "PNI Positive" in verdict:
            st.error(f"**{verdict}**")
        elif "PNI Negative" in verdict:
            st.success(f"**{verdict}**")
        else:
            st.info(f"**{verdict}**")

        # Annotated image
        col1, col2 = st.columns([2, 1])

        with col1:
            st.image(annotated, caption="Detection Results", use_container_width=True)

        with col2:
            if regions:
                df = pd.DataFrame([
                    {
                        "Region": f"R{r['region_id']}",
                        "Nerve Conf.": f"{r['nerve_prob']:.1%}",
                        "PNI Prob.": f"{r['pni_prob']:.1%}",
                        "PNI Status": "🔴 POSITIVE" if r["pni_positive"] else "🟢 Negative",
                        "Patches": r["n_patches"],
                    }
                    for r in regions
                ])
                st.dataframe(df, use_container_width=True, hide_index=True)
            else:
                st.write("No nerve regions detected.")

        # Store results in session
        st.session_state["last_verdict"] = verdict

elif uploaded_file is None and "example_image" not in st.session_state:
    st.info("👆 Upload an image or select an example above to get started.")
