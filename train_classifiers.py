"""One-time script to train and pickle the nerve/PNI classifiers.

Run locally before deploying to HuggingFace Spaces:
    python train_classifiers.py
"""

import numpy as np
import joblib
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


def main():
    features_path = Path(__file__).parent / "features.npz"
    out_dir = Path(__file__).parent / "classifiers"
    out_dir.mkdir(exist_ok=True)

    print("Loading features...")
    data = np.load(features_path, allow_pickle=True)
    features = data["features"]
    labels_nerve = data["labels_nerve"]
    labels_pni = data["labels_pni"]

    # --- Nerve detector ---
    print(f"Training nerve detector on {len(features)} patches...")
    nerve_clf = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            C=1.0, max_iter=2000, class_weight="balanced",
            solver="lbfgs", random_state=42,
        )),
    ])
    nerve_clf.fit(features, labels_nerve)
    nerve_path = out_dir / "nerve_clf.pkl"
    joblib.dump(nerve_clf, nerve_path)
    print(f"  Saved: {nerve_path} ({nerve_path.stat().st_size / 1024:.1f} KB)")

    # --- PNI classifier (nerve patches only) ---
    pni_mask = labels_pni != -1
    print(f"Training PNI classifier on {pni_mask.sum()} nerve patches...")
    pni_clf = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            C=1.0, max_iter=2000, class_weight="balanced",
            solver="lbfgs", random_state=42,
        )),
    ])
    pni_clf.fit(features[pni_mask], labels_pni[pni_mask])
    pni_path = out_dir / "pni_clf.pkl"
    joblib.dump(pni_clf, pni_path)
    print(f"  Saved: {pni_path} ({pni_path.stat().st_size / 1024:.1f} KB)")

    print("\nDone! Classifiers ready for deployment.")


if __name__ == "__main__":
    main()
