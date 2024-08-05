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

SEED = 1234
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
os.environ['CUDA_VISIBLE_DEVICE'] = "0"

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def train(model, trainloader, optimizer, criterion):
    model.train()
    #print('Training')
    train_running_loss = 0.0
    train_running_correct = 0
    counter = 0
    merged_tensor = torch.zeros(1,)
    
    for i, data in tqdm(enumerate(trainloader), total=len(trainloader)):
        counter += 1
        image, labels = data
        image = image.to(device)
        labels = labels.to(device)
        merged_tensor = merged_tensor.to(device)
        optimizer.zero_grad()
        outputs = model(image)
                        
        loss = criterion(outputs, labels)
        train_running_loss += loss.item()
        _, preds = torch.max(outputs.data, 1)
        train_running_correct += (preds == labels).sum().item()
        loss.backward()
        optimizer.step()
    
    epoch_loss = train_running_loss / counter
    epoch_acc = 100. * (train_running_correct / len(trainloader.dataset))
    return epoch_loss, epoch_acc

def validate(model, testloader, criterion):
    model.eval()
    #print('Validation')
    valid_running_loss = 0.0
    valid_running_correct = 0
    counter = 0

    with torch.no_grad():
        for i, data in tqdm(enumerate(testloader), total=len(testloader)):
            counter += 1
            
            image, labels = data
            image = image.to(device)
            labels = labels.to(device)
            outputs = model(image)
                                
            loss = criterion(outputs, labels)
            valid_running_loss += loss.item()
            _, preds = torch.max(outputs.data, 1)
            valid_running_correct += (preds == labels).sum().item()
        
    epoch_loss = valid_running_loss / counter
    epoch_acc = 100. * (valid_running_correct / len(testloader.dataset))
    return epoch_loss, epoch_acc

def plot_loss_curve(train_loss, valid_loss):
    plt.plot(train_loss, label='train loss')
    plt.plot(valid_loss, label='validation loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss curve')
    plt.legend()
    plt.savefig('/home/ldap_howard/script/loss_curve_resnet18_0307_2.png')
    plt.close()

def plot_acc_curve(train_acc, valid_acc):
    plt.plot(train_acc, label='train acc')
    plt.plot(valid_acc, label='validation acc')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Accuracy curve')
    plt.legend()
    plt.savefig('/home/ldap_howard/script/acc_curve_resnet18_0307_2.png')
    plt.close()


if __name__ == '__main__':
    train_dir = '/LVM_data/ldap_howard/NCT-CRC-HE-100K/'
    pretrained_size = 224
    pretrained_means = [0.485, 0.456, 0.406]
    pretrained_stds= [0.229, 0.224, 0.225]

    train_transforms = transforms.Compose([
                            transforms.Resize(pretrained_size),
                            transforms.RandomRotation(5),
                            transforms.RandomHorizontalFlip(0.5),
                            transforms.RandomVerticalFlip(0.5),
                            transforms.RandomCrop(pretrained_size, padding = 10),
                            transforms.ToTensor(),
                            transforms.Normalize(mean = pretrained_means, 
                                                    std = pretrained_stds)
                        ])

    test_transforms = transforms.Compose([
                            transforms.Resize(pretrained_size),
                            transforms.ToTensor(),
                            transforms.Normalize(mean = pretrained_means, 
                                                    std = pretrained_stds)
                        ])
    train_data = datasets.ImageFolder(root = train_dir, transform = train_transforms)
    
    VALID_RATIO = 0.8
    #TEST_RATIO = 0.5
    n_train_examples = int(len(train_data) * VALID_RATIO)
    n_valid_examples = len(train_data) - n_train_examples
    train_data, valid_data = data.random_split(train_data, [n_train_examples, n_valid_examples])
    
    valid_data = copy.deepcopy(valid_data)
    valid_data.dataset.transform = test_transforms    

    print(f'Number of training examples: {len(train_data)}')
    print(f'Number of validation examples: {len(valid_data)}')

    BATCH_SIZE = 256
    train_iterator = data.DataLoader(train_data, shuffle = True, batch_size = BATCH_SIZE)
    valid_iterator = data.DataLoader(valid_data, batch_size = BATCH_SIZE)

    model = models.resnet18(weights="IMAGENET1K_V1").cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.00005)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_loss, valid_loss = [], []
    train_acc, valid_acc = [], []
    min_loss = np.inf
    epochs = 150
    path = '/home/ldap_howard/script/resnet18_0307_2_summary.txt'

    for epoch in range(epochs):
        print(f"[INFO]: Epoch {epoch+1} of {epochs}")
        
        train_epoch_loss, train_epoch_acc = train(model, train_iterator, 
                                                optimizer, criterion)
        valid_epoch_loss, valid_epoch_acc = validate(model, valid_iterator,  
                                                    criterion)
        train_loss.append(train_epoch_loss)
        valid_loss.append(valid_epoch_loss)
        train_acc.append(train_epoch_acc)
        valid_acc.append(valid_epoch_acc)
        
        if min_loss > valid_epoch_loss:
            with open(path, 'a') as f:
                f.write("save model!!!!!!!!!!!!!" + '\n')

            torch.save(model.state_dict(), 'resnet18NCT_0307_2.pt')
            min_loss = valid_epoch_loss

        with open(path, 'a') as f:
            f.write('Epoch: '+ str(epoch) +'\n')
            f.write('Training loss: '+str(train_epoch_loss)+'\n')
            f.write('Training acc: '+str(train_epoch_acc)+'\n')
            f.write('Validation loss: '+str(valid_epoch_loss)+'\n')
            f.write('Validation acc: '+str(valid_epoch_acc)+'\n')
            f.write('-----------------------------------------------------'+'\n')

    plot_acc_curve(train_acc, valid_acc)
    plot_loss_curve(train_loss, valid_loss)

    model.load_state_dict(torch.load('resnet18NCT_0307_2.pt'))

    externel_test_data = datasets.ImageFolder(root = '/ORCA_lake/TCGA-COAD/CRC-VAL-HE-7K/', transform = test_transforms)
    externel_test_iterator = data.DataLoader(externel_test_data, batch_size = BATCH_SIZE)
    model.load_state_dict(torch.load('resnet18NCT_0307_2.pt'))
    externel_test_loss, externel_test_acc = validate(model, externel_test_iterator, criterion)

    with open(path, 'a') as f:
        f.write('External test' +'\n')
        f.write('Externel test loss: '+str(externel_test_loss)+'\n')
        f.write('Externel test acc: '+str(externel_test_acc)+'\n')


