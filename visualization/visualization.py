import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from torch.utils.data import Dataset, ConcatDataset
from pathlib import Path
from typing import Optional, Sequence
from PIL import Image, ImageDraw
import openslide
import PIL
import re
import os
import random
import numpy as np
import pandas as pd
import tqdm
from RetCLL import ResNet
import matplotlib.pyplot as plt
import seaborn as sns
import math

SEED = 1234
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

class FocalLoss(nn.Module):
    
    def __init__(self, weight=None, 
                 gamma=2., reduction='mean'):
        nn.Module.__init__(self)
        self.weight = weight
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, logits, label):
        log_prob = F.log_softmax(logits, dim=-1)
        prob = torch.exp(log_prob)
        return F.nll_loss(
            ((1 - prob) ** self.gamma) * log_prob, 
            label, 
            weight=self.weight,
            reduction = self.reduction
        )

class Attention(nn.Module):
    def __init__(self, L=2048, D=1024, dropout=True, n_classes=2, top_k=1):
        super(Attention, self).__init__()
        self.L = L
        self.D = D
        self.K = 1

        self.layer1 = nn.Linear(self.L, self.D)
        if dropout:
            self.attention_V = nn.Sequential(nn.Linear(self.D, 256), nn.Tanh(), nn.Dropout(0.25))
            self.attention_U = nn.Sequential(nn.Linear(self.D, 256), nn.Sigmoid(), nn.Dropout(0.25))
        else:
            self.attention_V = nn.Sequential(nn.Linear(self.D, 256), nn.Tanh())
            self.attention_U = nn.Sequential(nn.Linear(self.D, 256), nn.Sigmoid())

        self.attention_weights = nn.Linear(256, self.K)

        self.classifier = nn.Sequential(nn.Linear(self.D, 512),
                                        nn.ReLU(),
                                        nn.Dropout(0.25),
                                        nn.Linear(512, 256),
                                        nn.ReLU(),
                                        nn.Linear(256,2),
                                        nn.Sigmoid())

        self.top_k = top_k

    def forward(self, x):
        x = self.layer1(x)
        A_V = self.attention_V(x)  # NxD
        A_U = self.attention_U(x)
        A = self.attention_weights(A_U* A_V)
        A = torch.transpose(A, 1, 0)  # KxN
        A1 = F.softmax(A, dim=1)  # softmax over N
        M = torch.mm(A1, x)  # KxL

        logits = self.classifier(M)
        y_probs = F.softmax(logits, dim=1)
        top_instance_idx = torch.topk(y_probs[:, 1], self.top_k, dim=0)[1].view(1, )
        top_instance = torch.index_select(logits, dim=0, index=top_instance_idx)
        Y_hat = torch.topk(top_instance, 1, dim=1)[1]
        Y_prob = F.softmax(top_instance, dim=1)
        results_dict = {}
        results_dict.update({'logits': top_instance, 'Y_prob': Y_prob, 'Y_hat': Y_hat, 'A': A})
        return A1

    def relocate(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.classifier.to(device)
        self.attention_V.to(device)
        self.attention_U.to(device)
        self.attention_weights.to(device)

class Attention_clam(nn.Module):
    def __init__(self, L=2048, D=1024, dropout=True, n_classes=2, top_k=1, instance_loss_fn=nn.CrossEntropyLoss()):
        super(Attention_clam, self).__init__()
        self.L = L
        self.D = D
        self.K = 1

        self.layer1 = nn.Linear(self.L, self.D)
        if dropout:
            self.attention_V = nn.Sequential(nn.Linear(self.D, 512), nn.Tanh(), nn.Dropout(0.25))
            self.attention_U = nn.Sequential(nn.Linear(self.D, 512), nn.Sigmoid(), nn.Dropout(0.25))
        else:
            self.attention_V = nn.Sequential(nn.Linear(self.D, 512), nn.Tanh())
            self.attention_U = nn.Sequential(nn.Linear(self.D, 512), nn.Sigmoid())

        self.attention_weights = nn.Linear(512, self.K)

        self.classifier = nn.Sequential(nn.Linear(1028, 1024),
                                        nn.ReLU(),
                                        nn.Dropout(0.25),
                                        nn.Linear(1024, 512),
                                        nn.ReLU(), 
                                        nn.Linear(512,4),
                                        nn.Sigmoid())
        self.top_k = top_k
        self.instance_loss = instance_loss_fn
        self.fc_c1 = nn.Sequential(nn.Linear(4, 4), nn.ReLU())
        self.fc_X = nn.Sequential(nn.Linear(1, 4), nn.Sigmoid())
        #self.fc_c = nn.Sequential(nn.Linear(4, 2), nn.Sigmoid())


    def relocate(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.classifier.to(device)
        self.attention_V.to(device)
        self.attention_U.to(device)
        self.attention_weights.to(device)
        self.instance_loss.to(device)
        self.inst_eval.to(device)
        self.fc_c.to(device)
        self.fc_X.to(device)
    
    def inst_eval(self, A_T, count):
        logits_c = self.fc_c1(count)
        hover_logits = torch.mm(A_T, logits_c)
        y_probs_c = F.softmax(logits_c, dim=1)
        #k = math.ceil(logits_c.size()[0] / 20)
        _, predicted_class = torch.max(y_probs_c, dim=1)
        predicted_prob = y_probs_c[torch.arange(y_probs_c.size(0)), predicted_class]
        top_instance_idx = torch.topk(predicted_prob, 5, largest=True)[1]
        top_instance = torch.index_select(y_probs_c, dim=0, index=top_instance_idx)
        _, pseudo_targets = torch.max(top_instance, dim=1)

        A_T = torch.transpose(A_T, 1, 0)  # KxN
        logits_x = self.fc_X(A_T)
        y_probs_x = F.softmax(logits_x, dim=1)
        top_instance_x = torch.index_select(logits_x, dim=0, index=top_instance_idx)
        pseudo_logits = top_instance_x

        return pseudo_logits, pseudo_targets, hover_logits
    

    def forward(self, x, count):
        x = self.layer1(x)
        A_V = self.attention_V(x)  # NxD
        A_U = self.attention_U(x)
        A = self.attention_weights(A_V*A_U)
        A_T = torch.transpose(A, 1, 0)  # KxN
        A_T = F.softmax(A_T, dim=1)  # softmax over N
        M = torch.mm(A_T, x)  # KxL

        pseudo_logits, pseudo_targets, hover_logits = self.inst_eval(A_T, count)
        instance_loss = self.instance_loss(pseudo_logits, pseudo_targets)

        M  = torch.cat((M, hover_logits), dim=1)
       
        logits = self.classifier(M)
        y_probs = F.softmax(logits, dim=1)
        top_instance_idx = torch.topk(y_probs[:, 1], self.top_k, dim=0)[1].view(1, )
        top_instance = torch.index_select(logits, dim=0, index=top_instance_idx)
        Y_hat = torch.topk(top_instance, 1, dim=1)[1]
        Y_prob = F.softmax(top_instance, dim=1)
        results_dict = {}
        results_dict.update({'logits': top_instance, 'Y_prob': Y_prob, 'Y_hat': Y_hat, 'Instance_loss': instance_loss, 'hover_logits':hover_logits})

        return A_T

def npy_loader(path):
    x = np.load(path,allow_pickle=True).item()
    x_im = torch.from_numpy(x['features'])
    return x_im

def npy_loader_count(path):
    x = np.load(path,allow_pickle=True).item()
    x_im = torch.from_numpy(x['counts'])
    return x_im

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Attention_clam().to(device)
    model.load_state_dict(torch.load('/home/ldap_howard/script/model/CMS_0307CV6_fold2_clam_checkpoint.pt'))

    test_dataset = pd.read_excel('/home/ldap_howard/script/CRC_CMS.xlsx', sheet_name='CMS_0604')
    test_x = test_dataset['Patients'].values
    test_y = test_dataset['status'].values  

    train_dir ='/ORCA_lake/TCGA-COAD/select_patches/CRC_DX_0307/'

    model.eval()
    with torch.no_grad():
        for slide_name, label in zip(test_x, test_y):
            print(slide_name)
            #slide_name = 'TCGA-CM-4744-01Z-00-DX1'
            npy_dir = slide_name + '.npy'
            data = npy_loader(os.path.join('/ORCA_lake/TCGA-COAD/feature/CRC_resnet0307/', npy_dir)).to(device)
            count = npy_loader_count(os.path.join('/ORCA_lake/TCGA-COAD/hovernet_kmeans/CRC_0307_MI2N/', npy_dir)).to(device)

            attention_score = model(data, count)
            max = torch.max(attention_score[0])
            min = torch.min(attention_score[0])

            svs_path = '/ORCA_lake/TCGA-COAD/wsi/CRC_svs/'+slide_name+'.svs'
            slide = openslide.open_slide(svs_path)
            original_width, original_height = slide.dimensions

            if 'aperio.AppMag' not in slide.properties: p =512
            elif slide.properties['aperio.AppMag'] == '40': p = 1024
            else: p = 512
            print(original_height// p, original_width//p)
            scores = np.zeros((original_height // p, original_width // p))
            
            input_image_path = '/ORCA_lake/TCGA-COAD/visualization/resized_image/CRC/'+slide_name+'.png'
            output_image_path = '/ORCA_lake/TCGA-COAD/visualization/CMS_0604_CV6/CRC/'+slide_name+'.png'

            patch_folder = os.path.join(train_dir, slide_name)
            patch_files = [file for file in os.listdir(patch_folder) if file.endswith('.png')]

            file_path = '/ORCA_lake/TCGA-COAD/patch_list/CRC_DX_0307/'+slide_name+'_list.txt'
            patches = pd.read_table(file_path)

            i = 0
            
            with open(file_path, "r") as file:
                for line in file:
                    patches = line.split()
                    for patch in patches:
                        pos1 = int(patch.split('_')[1].split('_')[0])
                        pos2 = int(patch.split('.')[0].split('_')[2])
                        scores[pos2][pos1] = attention_score[0][i]
                        i+=1
            
            scores[scores == 0] = np.nan
            
            resized_image = Image.open(input_image_path)
            image_width, image_height = resized_image.size
            array_width, array_height = scores.shape
            enlarged_scores = np.zeros((image_height, image_width))
            ratio = image_height // array_width
            for i in range(array_width):
                for j in range(array_height):
                    enlarged_scores[i*ratio:i*ratio+ratio, j*ratio:j*ratio+ratio] = scores[i, j]

            plt.imshow(enlarged_scores, vmin=min, vmax=max, cmap='Reds', interpolation='nearest', alpha=1.0)
            plt.imshow(resized_image, alpha=0.4)

            plt.axis('off')
            plt.savefig(output_image_path)
            plt.close()

            #break

            
