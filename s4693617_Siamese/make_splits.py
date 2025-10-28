# make_splits.py
# Create patient-level Stratified K-Fold splits for ISIC-2020.
# Produces per-fold train/val CSVs with columns: image_path,label,patient_id
# Usage example:
# python make_splits.py \
#   --train_csv data/raw/train.csv \
#   --images_dir data/raw/train \
#   --out_dir data/splits \
#   --n_splits 5 --seed 1337
# 
# Notes:
# - Expects train.csv to contain at least: image_name, patient_id, target
# - If image_path is not present, it will be constructed as images_dir/<image_name>.jpg
# - Stratification is done at the patient level (patient label = max(target) across that patient's images).
# - I will use 5 splits for the network at this stage, training 5 models against 4 folds, testing on the fifth to get accuracy metric.

from __future__ import annotations
import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

try:
    # Available in scikit-learn >= 1.1
    from sklearn.model_selection import StratifiedKFold
except Exception as e:
    print("[warn] scikit-learn missing or outdated; please install sklearn>=1.0", file=sys.stderr)
    raise


def infer_columns(df: pd.DataFrame):
    cols = {c.lower(): c for c in df.columns}
    # required
    image_col = cols.get("image_path") or cols.get("image_name") or cols.get("image")
    pid_col = cols.get("patient_id") or cols.get("patient") or cols.get("patientid")
    label_col = cols.get("target") or cols.get("label")
    if image_col is None:
        raise ValueError("Could not find an image column (image_path/image_name/image)")
    if pid_col is None:
        raise ValueError("Could not find a patient_id column (patient_id/patient/patientid)")
    if label_col is None:
        raise ValueError("Could not find a label/target column (target/label)")
    return image_col, pid_col, label_col


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_csv", required=True, help="Kaggle train.csv with image_name, patient_id, target")
    ap.add_argument("--images_dir", required=False, default=None, help="Directory containing images; used to build image_path if needed")
    ap.add_argument("--out_dir", required=True, help="Output directory for fold CSVs")
    ap.add_argument("--n_splits", type=int, default=5)
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.train_csv)
    image_col, pid_col, label_col = infer_columns(df)

    # Construct image_path if necessary
    if image_col.lower() != "image_path":
        if args.images_dir is None:
            raise ValueError("image_path not present; please pass --images_dir to construct file paths.")
        img_root = Path(args.images_dir)
        # ISIC-2020 images are typically JPG
        df["image_path"] = df[image_col].astype(str).apply(lambda s: str((img_root / f"{s}.jpg").as_posix()))
    else:
        df["image_path"] = df[image_col].astype(str)

    # Normalize column names expected by our training code
    df["label"] = df[label_col].astype(int)
    df["patient_id"] = df[pid_col].astype(str)

    # Patient-level label: any melanoma within the patient ⇒ patient_label=1
    patient_labels = df.groupby("patient_id")["label"].max().reset_index().rename(columns={"label": "patient_label"})

    # Stratified K-Fold over patients
    skf = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)
    patients = patient_labels["patient_id"].values
    y = patient_labels["patient_label"].values

    for fold, (train_idx, val_idx) in enumerate(skf.split(patients, y)):
        train_pids = set(patients[train_idx])
        val_pids = set(patients[val_idx])

        tr = df[df["patient_id"].isin(train_pids)][["image_path", "label", "patient_id"]].reset_index(drop=True)
        va = df[df["patient_id"].isin(val_pids)][["image_path", "label", "patient_id"]].reset_index(drop=True)

        tr_out = out_dir / f"train_fold{fold}.csv"
        va_out = out_dir / f"val_fold{fold}.csv"
        tr.to_csv(tr_out, index=False)
        va.to_csv(va_out, index=False)
        print(f"[fold {fold}] train={len(tr)} val={len(va)} → {tr_out.name}, {va_out.name}")

    # Also emit a convenience single split (fold 0)
    (out_dir / "train.csv").write_text((out_dir / "train_fold0.csv").read_text())
    (out_dir / "val.csv").write_text((out_dir / "val_fold0.csv").read_text())

    # Summary
    counts = patient_labels["patient_label"].value_counts().to_dict()
    print(f"Patients: total={len(patient_labels)}, pos={counts.get(1,0)}, neg={counts.get(0,0)}")
    print(f"Wrote {args.n_splits} folds into {out_dir}")


if __name__ == "__main__":
    main()
