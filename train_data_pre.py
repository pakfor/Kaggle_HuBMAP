# -*- coding: utf-8 -*-
"""
Created on Sat May 27 08:20:57 2023

@author: NGPF
"""
import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

save_output_dir = "H:/Kaggle Competitions/HuBMAP/hubmap-hacking-the-human-vasculature/train_prep"
hist_img_dir = "H:/Kaggle Competitions/HuBMAP/hubmap-hacking-the-human-vasculature/train_stain_norm"
blood_vessel_mask_dir = "H:/Kaggle Competitions/HuBMAP/hubmap-hacking-the-human-vasculature/train_mask/Blood Vessel/PNG"
masked_hist_img_list = os.listdir(blood_vessel_mask_dir)

x_arr = np.zeros((len(masked_hist_img_list), 512, 512, 3), dtype = 'float16')
y_arr = np.zeros((len(masked_hist_img_list), 512, 512, 1), dtype = 'float16')

for i in tqdm(range(0, len(masked_hist_img_list))):
    hist_img = plt.imread(f"{hist_img_dir}/{masked_hist_img_list[i]}")[:, :, 0:3]
    x_arr[i, :, :, :] = np.reshape(hist_img, (512, 512, 3))
    hist_img_mask = plt.imread(f"{blood_vessel_mask_dir}/{masked_hist_img_list[i]}")[:, :, 0]
    y_arr[i, :, :, :] = np.reshape(hist_img_mask, (512, 512, 1))

nr_train = 1200
nr_test = 433

np.save(f"{save_output_dir}/x_train.npy", x_arr[0:nr_train, :, :, :])
np.save(f"{save_output_dir}/y_train.npy", y_arr[0:nr_train, :, :, :])
np.save(f"{save_output_dir}/x_test.npy", x_arr[nr_train:, :, :, :])
np.save(f"{save_output_dir}/y_test.npy", y_arr[nr_train:, :, :, :])