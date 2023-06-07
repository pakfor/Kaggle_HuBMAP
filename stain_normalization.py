# -*- coding: utf-8 -*-
"""
Created on Fri May 26 21:01:55 2023

@author: NGPF
"""
import os
import staintools
from tqdm import tqdm
import matplotlib.pyplot as plt

def stain_norm(stain_ref, image):
    target = staintools.LuminosityStandardizer.standardize(stain_ref)
    to_transform = staintools.LuminosityStandardizer.standardize(image)
    normalizer = staintools.StainNormalizer(method='vahadane')
    normalizer.fit(target)
    transformed = normalizer.transform(to_transform)
    return transformed

hist_img_dir = "H:/Kaggle Competitions/HuBMAP/hubmap-hacking-the-human-vasculature/train"
hist_img_list = os.listdir(hist_img_dir)
stain_ref_dir = "H:/Kaggle Competitions/HuBMAP/hubmap-hacking-the-human-vasculature/stain_ref.tif"
stain_ref_img = plt.imread(stain_ref_dir)
hist_img_norm_save_dir = "H:/Kaggle Competitions/HuBMAP/hubmap-hacking-the-human-vasculature/train_stain_norm"

for each_img in tqdm(hist_img_list):
    hist_img = plt.imread(f"{hist_img_dir}/{each_img}")
    hist_img_norm = stain_norm(stain_ref = stain_ref_img, image = hist_img)
    plt.imsave(f"{hist_img_norm_save_dir}/{each_img.replace('tif', 'png')}", hist_img_norm)