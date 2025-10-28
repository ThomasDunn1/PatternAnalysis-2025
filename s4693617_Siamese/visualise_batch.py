from pathlib import Path
import torch
import torchvision.utils as vutils
from dataset import ISIC2020Dataset

out = Path("runs"); out.mkdir(parents=True, exist_ok=True)
ds = ISIC2020Dataset("data/splits/train_fold0.csv", image_size=224, mode="train")

# pick first 16 safely
n = min(16, len(ds))
imgs = torch.stack([ds[i][0] for i in range(n)], dim=0)  # [N,3,H,W]
vutils.save_image(imgs, out / "sample_batch.png", nrow=4)
print("Wrote", (out / "sample_batch.png").as_posix())