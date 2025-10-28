from pathlib import Path
import pandas as pd

"""
Code to filter duplicates from the train.csv raw data file made by the build_train_csv.py. Uses duplicate_image_list.csv saved from the ISIC website. 
"""

RAW = Path("data/raw")

train = pd.read_csv(RAW / "train.csv")
dups  = pd.read_csv(RAW / "duplicate_image_list.csv")  # adjust filename if needed

# normalize column names
train.columns = [c.lower() for c in train.columns]
dups.columns  = [c.lower() for c in dups.columns]

# extract a flat set of duplicate image names (stems like ISIC_12345)
dup_names = set()
if {"image1", "image2"}.issubset(dups.columns):
    dup_names.update(dups["image1"].astype(str))
    dup_names.update(dups["image2"].astype(str))
else:
    # find any column that looks like image name(s)
    cand_cols = [c for c in dups.columns if "image" in c]
    if not cand_cols:
        raise ValueError("Couldn't find image columns in duplicate list CSV.")
    for c in cand_cols:
        dup_names.update(dups[c].dropna().astype(str))

before = len(train)
train = train[~train["image_name"].isin(dup_names)].reset_index(drop=True)
after = len(train)
print(f"Removed {before - after} duplicates; remaining {after}")

train.to_csv(RAW / "train.csv", index=False)
print("Wrote deduplicated data/raw/train.csv")
