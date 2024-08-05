import os
import openslide
from openslide import open_slide
from skimage import io
import numpy as np
from tqdm import tqdm
import pandas as pd
import time
import argparse
from PIL import Image
import multiprocessing
import sys
from scipy.stats import entropy

num_cores = 48
def cal_entropy(image_array):
    _bins = 256
    hist, _ = np.histogram(image_array.ravel(), bins=_bins, range=(0, _bins))
    prob_dist = hist / hist.sum()
    image_entropy = entropy(prob_dist, base=2)
    return image_entropy

def Segmentation_main(svs_dir,img_dir,threshold,patch_size,target_mag, n):
    slide_name = n+'.svs'

    slidePath = '{:s}{:s}'.format(svs_dir, slide_name)
    slide = open_slide(slidePath)

    if 'aperio.AppMag' not in slide.properties:
        print('no magnification param')
        magnification = 20
        print(slide)
    else:
        magnification = float(slide.properties['aperio.AppMag'])

    extract_patch_size = int(patch_size * magnification / target_mag)
    scale = 20.0 / patch_size

    w, h = slide.level_dimensions[0]
    w = w // extract_patch_size * extract_patch_size
    h = h // extract_patch_size * extract_patch_size

    x, y = 0, 0
    for i in range(0, w, extract_patch_size):
        for j in range(0, h, extract_patch_size):
            patch = slide.read_region((i, j), level=0, size=[extract_patch_size, extract_patch_size])

            # downsample to target patch size
            patch = patch.resize([patch_size, patch_size])

            # check if the patch contains tissue
            gray_image = patch.convert("L")
            image_array = np.array(gray_image)
            if np.sum(image_array < 200) >= threshold and cal_entropy(image_array) > 6.5:
                name = 'region_'+str(x)+'_'+str(y)
                patch_name = '{:s}/{:s}.png'.format(img_dir, name)
                patch.save(patch_name)

            y+=1
        y=0
        x+=1

def task_function(n):
    svs_dir = os.path.join('/lake_data/TCGA-COAD/wsi/READ_svs/')
    img_dir = os.path.join('/lake_data/TCGA-COAD/png/READ_20/', n)
    threshold = 512 * 512 / 2  # threshod to remove blank patches
    patch_size = 512    # the size of patches extracted from 20x svs image
    target_mag = 20

    if not os.path.exists(img_dir):
        os.mkdir(img_dir)
    Segmentation_main(svs_dir, img_dir, threshold, patch_size, target_mag, n)

if __name__ == "__main__":
    pool = multiprocessing.Pool(processes=num_cores)
    list = []
    filename = sys.argv[1]
    with open(filename, 'r') as f:
         for line in f:
             line = line.strip()
             filename = line.split('.')[0]
             list.append(filename)

    pool.map(task_function,list)
    pool.close()
    pool.join()
