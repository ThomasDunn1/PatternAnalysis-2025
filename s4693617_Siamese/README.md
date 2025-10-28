# COMP3070 Problem 9: ISIC-2020 Siamese Network to 0.8 accuracy
Author: Thomas Dunn UQ ID: s4693617

This repo will train a Siamese-style model (pretrained CNN backbone + metric loss) on the ISIC-2020 dataset for melanoma vs normal.
This initial commit sets up the environment, directories, ignore rules, and dependencies.

## Status
- ✅ Repo skeleton & Python virtual environment
- ✅ `.gitignore` for caches, venvs, runs, and raw data
- ✅ `requirements.txt` with core libs
- ✅ Data splits (`make_splits.py`)
- ✅ Dataset/dataloader (`dataset.py`)
- ✅ Model modules (`modules.py`)
- ✅ Losses modules (`losses.py`)
- ✅ Training script (`train.py`)
- ☐ Prediction example (`predict.py`)

## Directory layout
s4693617_Siamese/  
├─ data/  
│ ├─ raw/ # Kaggle images & original CSVs  
│ └─ splits/ # generated patient-level split CSVs  
├─ runs/ # logs, plots, checkpoints  
├─ .venv/ # Python virtualenv  
├─ .gitignore  
├─ requirements.txt  
└─ README.md  

## Environment
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Sample Batch grid
Generated from visualise_batch.py:

<p align="center">
  <img src="runs/sample_batch.png" width="320" alt="Sample batch (train transforms)">
</p>
<sub><em>Figure: 4×4 grid of augmented training images (patient-wise splits).</em></sub>


## Next Steps
Checkpointing & Best-Model Selection
- Save best.pt whenever val_auc improves; persist config (backbone, embed_dim, image_size).
- At end, reload best.pt and run a final eval on the val fold (placeholder for a future test set).
- Write a test_stats.txt with metrics and the chosen threshold.