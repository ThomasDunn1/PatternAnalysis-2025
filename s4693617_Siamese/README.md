# COMP3070 Problem 9: ISIC-2020 Siamese Network to 0.8 accuracy

This repo will train a Siamese-style model (pretrained CNN backbone + metric loss) on the ISIC-2020 dataset for melanoma vs normal.
This initial commit sets up the environment, directories, ignore rules, and dependencies.

## Status (Step 0)
- ✅ Repo skeleton & Python virtual environment
- ✅ `.gitignore` for caches, venvs, runs, and raw data
- ✅ `requirements.txt` with core libs
- ☐ Data splits (`make_splits.py`)
- ☐ Dataset/dataloader (`dataset.py`)
- ☐ Model modules (`modules.py`)
- ☐ Training script (`train.py`)
- ☐ Prediction example (`predict.py`)

## Directory layout
isic2020-snn/
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


## Next Steps
1. Add make_splits.py to generate patient-level stratified folds.
2. Implement dataset.py with transforms and a simple loader smoke test.
3. Add modules.py (pretrained backbone + projection head).
4. Implement metric loss & minimal train loop; then validation/plots/checkpoints.