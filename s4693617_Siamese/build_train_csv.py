# build_train_csv.py
import pandas as pd
from pathlib import Path

RAW = Path("data/raw")

gt = pd.read_csv(RAW / "training_ground_truth.csv")
meta = pd.read_csv(RAW / "training_metadata_v2.csv")

# Normalize column names
gt.columns = [c.lower() for c in gt.columns]
meta.columns = [c.lower() for c in meta.columns]

# Derive 'target' if it's not already present (1 = malignant/melanoma, 0 = benign/normal)
if "target" not in gt.columns:
    if "benign_malignant" in gt.columns:
        gt["target"] = (gt["benign_malignant"].astype(str).str.lower().isin(["malignant","melanoma"])).astype(int)
    elif "diagnosis" in gt.columns:
        gt["target"] = (gt["diagnosis"].astype(str).str.lower().str.contains("melanoma")).astype(int)
    else:
        raise ValueError("Ground truth CSV lacks 'target', 'benign_malignant', or 'diagnosis'.")

# Keep only the needed columns
gt_small   = gt[["image_name","target"]]
meta_small = meta[["image_name","patient_id"]]

# Merge
train = gt_small.merge(meta_small, on="image_name", how="inner")

# Sanity checks
assert train["image_name"].is_unique, "image_name not unique after merge"
assert {"image_name","patient_id","target"}.issubset(train.columns)

# Write Kaggle-style train.csv
out = RAW / "train.csv"
train.to_csv(out, index=False)
print(f"Wrote {out} with shape {train.shape}")
