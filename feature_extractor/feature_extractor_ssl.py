from pathlib import Path
import torch
import torch.nn as nn
import os
import re
from typing import Optional, Sequence
import torch
from torch.utils.data import Dataset, ConcatDataset
import PIL
from torchvision import transforms
import numpy as np
import random
from tqdm import tqdm
import ResNet

SEED = 1234
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

os.environ['CUDA_VISIBLE_DEVICE'] = "0"


class SlideTileDataset(Dataset):
    def __init__(self, slide_dir: Path, transform=None, *, repetitions: int = 1) -> None:
        self.tiles = list(slide_dir.glob("*.png"))
        assert self.tiles, f"no tiles found in {slide_dir}"
        self.tiles *= repetitions
        self.transform = transform

    def __len__(self):
        return len(self.tiles)

    def __getitem__(self, i):
        image = PIL.Image.open(self.tiles[i])
        patch = str(self.tiles[i]).split('/')[6]
        if self.transform:
            image = self.transform(image)
        
        return image, patch

    def _get_coords(filename) -> Optional[np.ndarray]:
        if matches := re.match(r".*\((\d+),(\d+)\)\.png", str(filename)):
            coords = tuple(map(int, matches.groups()))
            assert len(coords) == 2, "Error extracting coordinates"
            return np.array(coords)
        else:
            return None


normal_transform = transforms.Compose(
    [
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)
augmenting_transform = transforms.Compose(
    [
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomApply([transforms.GaussianBlur(3)], p=0.5),
        transforms.RandomApply(
            [
                transforms.ColorJitter(
                    brightness=0.1, contrast=0.2, saturation=0.25, hue=0.125
                )
            ],
            p=0.5,
        ),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)


if __name__ == "__main__":
    model = ResNet.resnet50(
        num_classes=128, mlp=False, two_branch=False, normlinear=True
    ).cuda()
    pretext_model = torch.load('/home/ldap_howard/script/best_ckpt.pth')
    model.fc = nn.Identity()
    model.load_state_dict(pretext_model, strict=True)

    outdir = '/ORCA_lake/TCGA-COAD/feature/CRC_DX_0307/'
    #outdir = '/ORCA_lake/TCGA-COAD/patch_list/PAIP_0307_IN/'
    if os.path.exists(outdir) == False: os.makedirs(outdir)
    file_path = '/ORCA_lake/TCGA-COAD/select_patches/CRC_DX_0307/'
    #file_path = '/LVM_data/ldap_howard/CRC_DX_0307/'
    slide_tile_paths = os.listdir(file_path)
    model.eval()
    with torch.no_grad():
        for slide_name in slide_tile_paths:
            slide_tile_path = os.path.join(file_path, slide_name)
            path1 = os.path.join(outdir, slide_name)
            #print(path1)
            if os.path.exists(path1 + '.npy'): continue
            slide_tile_path = Path(slide_tile_path)

            unaugmented_ds = SlideTileDataset(slide_tile_path, normal_transform)
            augmented_ds = SlideTileDataset(
                slide_tile_path, augmenting_transform, repetitions=0
            )
            ds = ConcatDataset([unaugmented_ds, augmented_ds])
            dl = torch.utils.data.DataLoader(ds, batch_size=64, shuffle=False, num_workers=os.cpu_count(), drop_last=False)

            feats = []
            patches = []
            for batch in tqdm(dl, leave=False):
                feats.append(model(batch[0].type_as(next(model.parameters()))).cpu().detach())

            feature_list = torch.cat(feats, dim=0)
            np.save('{:s}{:s}.npy'.format(outdir, slide_name), {'features': feature_list.numpy()})

            #save_path = outdir + slide_name + "_list.txt"
            #with open(save_path, "w") as file:
            #    for item in patches:
            #        file.write(str(item) + "\n")

        
