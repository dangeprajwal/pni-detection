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
    page_title="PNI Detection",
    page_icon="🔬",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ── Mobile-friendly CSS ──────────────────────────────────────────────

st.markdown("""
<style>
    /* Tighter padding on mobile */
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 1rem;
        max-width: 800px;
    }
    /* Smaller title on mobile */
    @media (max-width: 640px) {
        h1 { font-size: 1.4rem !important; }
        .block-container { padding-left: 0.8rem; padding-right: 0.8rem; }
    }
    /* Make dataframe scroll horizontally */
    .stDataFrame { overflow-x: auto; }
    /* Full-width buttons */
    .stButton > button { width: 100%; }
</style>
""", unsafe_allow_html=True)

# ── Cached model loading (runs once) ─────────────────────────────────

@st.cache_resource(show_spinner="Loading Phikon-v2 foundation model... (first time takes ~60s)")
def load_model():
    import gc, os
    os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"

    device = "cuda" if torch.cuda.is_available() else (
        "mps" if torch.backends.mps.is_available() else "cpu"
    )

    # Use float32 on CPU (most compatible), float16 on GPU
    if device == "cpu":
        model = AutoModel.from_pretrained(
            "owkin/phikon-v2",
            trust_remote_code=True,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
        ).eval()
    else:
        model = AutoModel.from_pretrained(
            "owkin/phikon-v2",
            trust_remote_code=True,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        ).to(device).eval()

    processor = AutoImageProcessor.from_pretrained(
        "owkin/phikon-v2",
        trust_remote_code=True,
        use_fast=True,
    )
    gc.collect()
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

# ── Settings (in expander, not sidebar — better for mobile) ──────────

# Defaults
crop_top = 0
crop_bottom = 0
crop_left = 0
crop_right = 0
nerve_threshold = 0.70
pni_threshold = 0.50

# ── Main UI ──────────────────────────────────────────────────────────

st.title("🔬 AI-Based PNI Detection in Histopathology")

st.markdown("""
Perineural invasion (PNI) — the infiltration of tumour cells into nerves — is a
critical histopathological finding that influences staging, prognosis, and
treatment decisions across multiple cancer types. Manual detection of PNI on
H&E-stained slides is time-consuming, subjective, and prone to inter-observer
variability.

This tool uses **[Phikon-v2](https://huggingface.co/owkin/phikon-v2)**, a
state-of-the-art pathology foundation model pre-trained on **460 million**
histopathology image tiles, to automatically:

1. **Detect nerve structures** in the tissue
2. **Classify perineural invasion** for each detected nerve

Upload an H&E-stained histopathology image below to get started.
""")

st.markdown(
    "🟩 **Green boxes** = Nerve without PNI &nbsp;&nbsp;·&nbsp;&nbsp; "
    "🟥 **Red boxes** = Nerve with PNI"
)

col_m1, col_m2 = st.columns(2)
with col_m1:
    st.metric("Nerve Detection AUC", "0.999")
with col_m2:
    st.metric("PNI Classification AUC", "0.979")

# ── Image Upload ─────────────────────────────────────────────────────

uploaded_file = st.file_uploader(
    "Upload H&E Image",
    type=["jpg", "jpeg", "png", "tif", "tiff", "bmp"],
    help="Drag and drop or click to upload",
    label_visibility="collapsed",
)

# ── Example Images ───────────────────────────────────────────────────

example_dir = Path(__file__).parent / "examples"
if example_dir.exists() and not uploaded_file:
    st.markdown("**Or try an example:**")
    example_files = sorted(example_dir.glob("*.jpg"))
    if example_files:
        cols = st.columns(min(len(example_files), 3))
        for i, ex in enumerate(example_files):
            with cols[i % 3]:
                img = Image.open(ex)
                label = ex.stem.replace("_", " ").replace("nerve ", "").title()
                st.image(img, caption=label, use_container_width=True)
                if st.button("Use", key=f"ex_{i}"):
                    st.session_state["example_image"] = str(ex)
                    st.rerun()

# ── Advanced Settings (collapsible, below upload) ────────────────────

with st.expander("⚙️ Advanced Settings"):
    st.markdown("**Detection Thresholds**")
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

    st.markdown("**Microscope UI Crop**")
    st.caption("Remove scale bar or metadata overlay from edges.")
    c1, c2 = st.columns(2)
    with c1:
        crop_top = st.number_input("Top (px)", min_value=0, max_value=500, value=0)
        crop_left = st.number_input("Left (px)", min_value=0, max_value=500, value=0)
    with c2:
        crop_bottom = st.number_input("Bottom (px)", min_value=0, max_value=500, value=0)
        crop_right = st.number_input("Right (px)", min_value=0, max_value=500, value=0)

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
    if st.button("🔍 Analyze", type="primary", use_container_width=True):
        with st.spinner("Analyzing... this may take 10-30 seconds."):
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

        # ── Display Results (single-column, mobile-first) ────────

        st.divider()

        # Verdict banner
        if "PNI Positive" in verdict:
            st.error(f"**{verdict}**")
        elif "PNI Negative" in verdict:
            st.success(f"**{verdict}**")
        else:
            st.info(f"**{verdict}**")

        # Annotated image (full width)
        st.image(annotated, caption="Detection Results", use_container_width=True)

        # Region details table (below image, scrollable)
        if regions:
            st.markdown("**Region Details**")
            df = pd.DataFrame([
                {
                    "Region": f"R{r['region_id']}",
                    "Nerve": f"{r['nerve_prob']:.0%}",
                    "PNI": f"{r['pni_prob']:.0%}",
                    "Status": "POSITIVE" if r["pni_positive"] else "Negative",
                }
                for r in regions
            ])
            st.dataframe(df, use_container_width=True, hide_index=True)
        else:
            st.write("No nerve regions detected.")

elif uploaded_file is None and "example_image" not in st.session_state:
    st.info("Upload an image or select an example above to get started.")

# ── Footer ───────────────────────────────────────────────────────────

st.divider()
st.caption(
    "**Research Use Only** — Not validated for clinical diagnosis. "
    "Built with [Phikon-v2](https://huggingface.co/owkin/phikon-v2)."
)
