import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.optim.lr_scheduler import _LRScheduler
import torch.utils.data as data

import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

from sklearn import decomposition
from sklearn import manifold
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm

import copy
from collections import namedtuple
import os
import random
import shutil
import time
from pytorch_lightning import LightningModule


def load_model():
    model = models.resnet18(weights="IMAGENET1K_V1")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #model.fc = nn.Linear(in_features=512, out_features=9, bias=True).cuda()
    model.to(device)
    model.load_state_dict(torch.load('resnet18NCT_0307.pt'))

    return model

def get_predictions(model, dataloader):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    preds_all = torch.tensor([]).to(device)

    with torch.no_grad():
        for i, data in tqdm(enumerate(dataloader), total=len(dataloader)):
            image, labels = data
            image = image.to(device)
            outputs = model(image)
            _, preds = torch.max(outputs.data, 1)
            preds_all = torch.cat((preds_all, preds),0)
    return preds_all

def get_type(pred_labels):
    is_adi = (pred_labels == 0).sum().item()
    is_back = (pred_labels == 1).sum().item()
    is_deb = (pred_labels == 2).sum().item()
    is_lym = (pred_labels == 3).sum().item()
    is_muc = (pred_labels == 4).sum().item()
    is_mus = (pred_labels == 5).sum().item()
    is_norm = (pred_labels == 6).sum().item()
    is_str = (pred_labels == 7).sum().item()
    is_tum = (pred_labels == 8).sum().item()
    print(is_adi, is_back, is_deb, is_lym, is_muc, is_mus, is_norm, is_str, is_tum)


def assign_tumor(tumor_root, infer_data, pred_labels):
    i = 0
    if os.path.exists(tumor_root) == False: os.makedirs(tumor_root)
    for label1 in pred_labels:
        if label1 == 8:
            start_dir = infer_data.samples[i][0]
            slide_name = start_dir.split('/')[5]
            slide_path = os.path.join(tumor_root, slide_name)

            if not os.path.exists(slide_path):
                os.makedirs(slide_path)
            shutil.copy(start_dir, slide_path)
        i+=1

if __name__ == "__main__":
    pretrained_size = 224
    pretrained_means = [0.485, 0.456, 0.406]
    pretrained_stds= [0.229, 0.224, 0.225]

    test_transforms = transforms.Compose([
        transforms.Resize(pretrained_size),
        transforms.ToTensor(),
        transforms.Normalize(mean = pretrained_means, std = pretrained_stds)
    ])

    model = load_model()

    BATCH_SIZE = 256
    infer_dir = '/ORCA_lake/TCGA-COAD/png/CPTAC_norm/'
    #infer_dir = '/LVM_data/ldap_howard/PAIP/'
    infer_data = datasets.ImageFolder(root = infer_dir, transform = test_transforms)
    infer_iterator = data.DataLoader(infer_data, batch_size = BATCH_SIZE)

    pred_labels = get_predictions(model, infer_iterator)
    tumor_root = '/ORCA_lake/TCGA-COAD/resnet18NCT_0307/TUM/CPTAC_0307/'
    assign_tumor(tumor_root, infer_data, pred_labels)



