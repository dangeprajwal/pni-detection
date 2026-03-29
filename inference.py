"""Core inference pipeline for PNI detection.

Two-stage approach:
  Stage 1: Nerve detection (is a nerve present in this patch?)
  Stage 2: PNI classification (is perineural invasion present?)

Uses Phikon-v2 foundation model as a frozen feature extractor
with pre-trained scikit-learn classifiers on top.
"""

import cv2
import numpy as np
import torch
from PIL import Image


# ── Default parameters ──────────────────────────────────────────────────
SCALES = [(512, 256), (384, 192)]  # (patch_size, stride)
NERVE_THRESHOLD = 0.7
PNI_THRESHOLD = 0.5
MERGE_RADIUS = 350
BATCH_SIZE = 16
MAX_IMAGE_DIM = 3000  # cap very large images


def crop_ui_overlay(img, top=0, bottom=0, left=0, right=0):
    """Crop microscope UI overlay from image edges.

    Defaults to 0 (no crop) — other pathologists may have
    different microscope software with different overlays.
    """
    h, w = img.shape[:2]
    y1 = top
    y2 = h - bottom if bottom > 0 else h
    x1 = left
    x2 = w - right if right > 0 else w
    return img[y1:y2, x1:x2]


def preprocess_image(img, crop_top=0, crop_bottom=0, crop_left=0, crop_right=0):
    """Prepare an uploaded image for inference."""
    # Ensure RGB
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)

    # Crop UI overlay if requested
    if any([crop_top, crop_bottom, crop_left, crop_right]):
        img = crop_ui_overlay(img, crop_top, crop_bottom, crop_left, crop_right)

    # Cap very large images
    h, w = img.shape[:2]
    if max(h, w) > MAX_IMAGE_DIM:
        scale = MAX_IMAGE_DIM / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)),
                         interpolation=cv2.INTER_AREA)

    return img


def extract_multiscale_patches(img, scales=None):
    """Extract patches at multiple scales using sliding windows."""
    if scales is None:
        scales = SCALES

    h, w = img.shape[:2]
    patches = []

    for patch_size, stride in scales:
        if h < patch_size or w < patch_size:
            continue
        for y in range(0, h - patch_size + 1, stride):
            for x in range(0, w - patch_size + 1, stride):
                patch = img[y:y + patch_size, x:x + patch_size]
                if patch_size != 512:
                    patch = cv2.resize(patch, (512, 512),
                                       interpolation=cv2.INTER_LANCZOS4)
                patches.append({
                    "patch": patch,
                    "x": x, "y": y, "size": patch_size,
                    "cx": x + patch_size // 2,
                    "cy": y + patch_size // 2,
                })

    # Handle small images: single center crop
    if not patches and h >= 64 and w >= 64:
        patch = cv2.resize(img, (512, 512), interpolation=cv2.INTER_LANCZOS4)
        patches.append({
            "patch": patch,
            "x": 0, "y": 0, "size": min(h, w),
            "cx": w // 2, "cy": h // 2,
        })

    return patches


def extract_features(patches, model, processor, device):
    """Extract Phikon-v2 CLS token features from patches."""
    all_feats = []
    for i in range(0, len(patches), BATCH_SIZE):
        batch = [Image.fromarray(p["patch"]) for p in patches[i:i + BATCH_SIZE]]
        inputs = processor(images=batch, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.inference_mode():
            if device.startswith("cuda"):
                with torch.autocast("cuda"):
                    out = model(**inputs)
            else:
                out = model(**inputs)
            all_feats.append(out.last_hidden_state[:, 0].float().cpu().numpy())

    return np.concatenate(all_feats) if all_feats else np.empty((0, 1024))


def cluster_detections(candidates, merge_radius=MERGE_RADIUS):
    """Group nearby nerve detections into distinct regions."""
    if not candidates:
        return []

    candidates = sorted(candidates, key=lambda d: d["nerve_prob"], reverse=True)
    used = set()
    clusters = []

    for i, d in enumerate(candidates):
        if i in used:
            continue
        cluster = [d]
        used.add(i)
        for j in range(i + 1, len(candidates)):
            if j in used:
                continue
            dist = np.sqrt(
                (d["cx"] - candidates[j]["cx"]) ** 2 +
                (d["cy"] - candidates[j]["cy"]) ** 2
            )
            if dist < merge_radius:
                cluster.append(candidates[j])
                used.add(j)
        clusters.append(cluster)

    return clusters


def draw_annotations(img, regions):
    """Draw bounding boxes and labels on the image."""
    vis = img.copy()
    for r in regions:
        x, y, s = r["x"], r["y"], r["size"]
        is_pni = r["pni_positive"]
        color = (255, 0, 0) if is_pni else (0, 200, 0)
        label_text = "PNI+" if is_pni else "PNI-"

        # Draw box
        cv2.rectangle(vis, (x, y), (x + s, y + s), color, 3)

        # Label with background
        label = f"R{r['region_id']} {label_text} ({r['pni_prob']:.0%})"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.55
        thickness = 2
        (tw, th), _ = cv2.getTextSize(label, font, font_scale, thickness)
        cv2.rectangle(vis, (x, y - th - 10), (x + tw + 6, y), color, -1)
        cv2.putText(vis, label, (x + 3, y - 6), font, font_scale,
                    (255, 255, 255), thickness)

    return vis


def run_inference(
    image,
    model,
    processor,
    nerve_clf,
    pni_clf,
    device,
    crop_top=0,
    crop_bottom=0,
    crop_left=0,
    crop_right=0,
    nerve_threshold=NERVE_THRESHOLD,
    pni_threshold=PNI_THRESHOLD,
):
    """Full two-stage PNI detection pipeline.

    Args:
        image: RGB numpy array (H, W, 3)
        model: Phikon-v2 model
        processor: Phikon-v2 image processor
        nerve_clf: Pre-trained nerve detection classifier
        pni_clf: Pre-trained PNI classification classifier
        device: "cuda", "mps", or "cpu"
        crop_*: Pixels to crop from each edge (microscope UI removal)
        nerve_threshold: Minimum probability to consider a region as nerve
        pni_threshold: Minimum probability to classify as PNI-positive

    Returns:
        annotated_image: RGB numpy array with bounding boxes
        verdict: Human-readable summary string
        regions: List of dicts with per-region details
    """
    # Preprocess
    img = preprocess_image(image, crop_top, crop_bottom, crop_left, crop_right)

    if img.size == 0 or img.shape[0] < 32 or img.shape[1] < 32:
        return image, "Error: Image too small after cropping.", []

    # Extract multi-scale patches
    patches = extract_multiscale_patches(img)
    if not patches:
        return img, "Error: Could not extract patches from image.", []

    # Extract features
    feats = extract_features(patches, model, processor, device)
    if feats.shape[0] == 0:
        return img, "Error: Feature extraction failed.", []

    # Stage 1: Nerve detection
    nerve_probs = nerve_clf.predict_proba(feats)[:, 1]
    candidates = []
    for idx, p in enumerate(patches):
        if nerve_probs[idx] >= nerve_threshold:
            p["nerve_prob"] = float(nerve_probs[idx])
            p["feat"] = feats[idx]
            candidates.append(p)

    # Cluster nearby detections
    clusters = cluster_detections(candidates)

    # Stage 2: PNI classification per region
    regions = []
    for ci, cluster in enumerate(clusters):
        cluster_feats = np.array([d["feat"] for d in cluster])
        pni_probs = pni_clf.predict_proba(cluster_feats)[:, 1]
        best = max(cluster, key=lambda d: d["nerve_prob"])
        max_nerve = max(d["nerve_prob"] for d in cluster)
        max_pni = float(pni_probs.max())

        regions.append({
            "region_id": ci + 1,
            "x": best["x"],
            "y": best["y"],
            "size": best["size"],
            "nerve_prob": round(max_nerve, 4),
            "pni_prob": round(max_pni, 4),
            "pni_positive": max_pni >= pni_threshold,
            "n_patches": len(cluster),
        })

    # Generate verdict
    has_nerve = len(regions) > 0
    pni_count = sum(1 for r in regions if r["pni_positive"])

    if not has_nerve:
        verdict = "No nerve structures detected in this image."
    elif pni_count > 0:
        verdict = (
            f"Detected {len(regions)} nerve region(s). "
            f"{pni_count} region(s) show perineural invasion (PNI positive)."
        )
    else:
        verdict = (
            f"Detected {len(regions)} nerve region(s). "
            f"No perineural invasion identified (PNI negative)."
        )

    # Draw annotations
    annotated = draw_annotations(img, regions)

    return annotated, verdict, regions
