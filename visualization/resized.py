import torch
#from pathlib import Path
from PIL import Image, ImageDraw
import openslide
#import PIL
#import re
import os
import random
import numpy as np
import pandas as pd
#import tqdm
#import sys
import multiprocessing

SEED = 1234
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def task_function(n):
    svs_path = os.path.join('/ORCA_lake/TCGA-COAD/wsi/PAIP_svs/', n)
    slide = openslide.open_slide(svs_path)

    original_width, original_height = slide.dimensions
    new_width = original_width // 64
    new_height = original_height // 64
    original_image_size = (new_width, new_height)

    roi = slide.read_region((0,0),0,(original_width, original_height))
    resized_roi = roi.resize((new_width, new_height), Image.ANTIALIAS)
    slide_name = n.split('.')[0]
    save_path = '/ORCA_lake/TCGA-COAD/visualization/resized_image/'+slide_name+'.png'
    resized_roi.save(save_path)

if __name__ == '__main__':
    pool = multiprocessing.Pool(processes=12)
    svs_dir = '/ORCA_lake/TCGA-COAD/wsi/PAIP_svs/'
    slides_to_be_processed = os.listdir(svs_dir)

    pool.map(task_function,list)
    pool.close()
    pool.join()

