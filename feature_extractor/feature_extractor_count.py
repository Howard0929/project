from pathlib import Path
import torch
import torch.nn as nn
import os
import re
import random
from typing import Optional, Sequence
import torch
from torch.utils.data import Dataset, ConcatDataset
import PIL
from PIL import Image
from torchvision import transforms
import numpy as np
from tqdm import tqdm
import ResNet
import pandas as pd

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
pretrained_means = [0.485, 0.456, 0.406]
pretrained_stds= [0.229, 0.224, 0.225]

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

class SlideTileDataset(Dataset):
    def __init__(self, slide_dir: Path, transform=None, CRC_count=None, repetitions: int = 1) -> None:
        self.tiles = list(slide_dir.glob("*.png"))
        assert self.tiles, f"no tiles found in {slide_dir}"
        self.tiles *= repetitions
        self.transform = transform
        self.count = CRC_count

    def __len__(self):
        return len(self.tiles)

    def __getitem__(self, i):
        image = PIL.Image.open(self.tiles[i])
        cell_count = self.count[(self.count['Patients'] == str(self.tiles[i]).split("/")[5]) & (self.count['Patch'] == str(self.tiles[i]).split("/")[6].split(".")[0])].iloc[:,[2,3,4,6]].values
        cell_count = torch.FloatTensor(cell_count)
        patch = str(self.tiles[i]).split('/')[6]

        if self.transform:
            image = self.transform(image)
        
        return image, cell_count, patch

    def _get_coords(filename) -> Optional[np.ndarray]:
        if matches := re.match(r".*\((\d+),(\d+)\)\.png", str(filename)):
            coords = tuple(map(int, matches.groups()))
            assert len(coords) == 2, "Error extracting coordinates"
            return np.array(coords)
        else:
            return None

if __name__ == "__main__":
    train_dir = '/ORCA_lake/TCGA-COAD/select_patches/CPTAC_0307/'
    slides_to_be_processed = os.listdir(train_dir)
    count_dir = '/ORCA_lake/TCGA-COAD/hovernet_kmeans/CPTAC_0307_MI2N/'
    #outdir = '/ORCA_lake/TCGA-COAD/patch_list/CPTAC_0307/'
    if os.path.exists(count_dir) == False: os.makedirs(count_dir)
    CRC_count = pd.read_excel('/ORCA_lake/TCGA-COAD/hovernet/allqupath/CPTAC0307_patch_count.xlsx')

    model = ResNet.resnet50(
        num_classes=128, mlp=False, two_branch=False, normlinear=True
    ).cuda()
    pretext_model = torch.load('/home/ldap_howard/script/best_ckpt.pth')
    model.fc = nn.Identity()
    model.load_state_dict(pretext_model, strict=True)
    model.eval()
    with torch.no_grad():
        for slide_name in slides_to_be_processed:
            slide_tile_path = os.path.join(train_dir, slide_name)
            print(slide_name)
            if not os.listdir(slide_tile_path): continue
            slide_tile_path = Path(slide_tile_path)

            unaugmented_ds = SlideTileDataset(slide_tile_path, normal_transform, CRC_count)
            augmented_ds= SlideTileDataset(slide_tile_path, augmenting_transform, CRC_count, repetitions=0)

            ds = ConcatDataset([unaugmented_ds, augmented_ds])
            dl = torch.utils.data.DataLoader(ds, batch_size=64, shuffle=False, num_workers=os.cpu_count(), drop_last=False)

            feats = []
            counts = []
            patches = []

            for batch, cell_count, patch in tqdm(dl, leave=False):
                count = cell_count.squeeze(dim=1)
                #feats.append(model(batch.type_as(next(model.parameters()))).cpu().detach())
                counts.append(count)

            #feature_list = torch.cat(feats, dim=0)
            count_list = torch.cat(counts, dim=0)
            np.save('{:s}{:s}.npy'.format(count_dir, slide_name), {'counts': count_list.numpy()})

            #save_path = outdir + slide_name + "_list.txt"
            #with open(save_path, "w") as file:
            #    for item in patches:
            #        file.write(str(item) + "\n")

