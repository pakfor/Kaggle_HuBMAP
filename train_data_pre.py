# -*- coding: utf-8 -*-
"""
Created on Sat May 27 08:20:57 2023

@author: NGPF
"""
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

save_output_dir = "H:/Kaggle Competitions/HuBMAP/hubmap-hacking-the-human-vasculature/multi_class"
metadata_save_dir = f"{save_output_dir}/metadata.csv"

hist_img_dir = "H:/Kaggle Competitions/HuBMAP/hubmap-hacking-the-human-vasculature/train_stain_norm"
blood_vessel_mask_dir = "H:/Kaggle Competitions/HuBMAP/hubmap-hacking-the-human-vasculature/train_mask/blood_vessel/PNG"
glomerulus_mask_dir = "H:/Kaggle Competitions/HuBMAP/hubmap-hacking-the-human-vasculature/train_mask/glomerulus/PNG"
unsure_mask_dir = "H:/Kaggle Competitions/HuBMAP/hubmap-hacking-the-human-vasculature/train_mask/unsure/PNG"
masked_hist_img_list = os.listdir(blood_vessel_mask_dir)

x_arr = np.zeros((len(masked_hist_img_list), 512, 512, 3), dtype = 'float16')
y_arr = np.zeros((len(masked_hist_img_list), 512, 512, 3), dtype = 'float16')

for i in tqdm(range(0, len(masked_hist_img_list))):
    hist_img = plt.imread(f"{hist_img_dir}/{masked_hist_img_list[i]}")[:, :, 0:3]
    x_arr[i, :, :, :] = np.reshape(hist_img, (512, 512, 3))
    hist_img_mask_bv = plt.imread(f"{blood_vessel_mask_dir}/{masked_hist_img_list[i]}")[:, :, 0]
    hist_img_mask_gm = plt.imread(f"{glomerulus_mask_dir}/{masked_hist_img_list[i]}")[:, :, 0]
    hist_img_mask_us = plt.imread(f"{unsure_mask_dir}/{masked_hist_img_list[i]}")[:, :, 0]
    y_arr[i, :, :, 0] = np.reshape(hist_img_mask_bv, (512, 512))
    y_arr[i, :, :, 1] = np.reshape(hist_img_mask_gm, (512, 512))
    y_arr[i, :, :, 2] = np.reshape(hist_img_mask_us, (512, 512))

nr_train = 1200
nr_test = 433

metadata = {"image_no":[], "usage":[]}
for i in range(0, len(masked_hist_img_list)):
    metadata["image_no"].append(masked_hist_img_list[i].replace(".png", ""))
    if i < nr_train:
        metadata["usage"].append("TRAIN")
    else:
        metadata["usage"].append("TEST")
metadata_df = pd.DataFrame(data = metadata)
metadata_df.to_csv(metadata_save_dir, index=False)

np.save(f"{save_output_dir}/x_train.npy", x_arr[0:nr_train, :, :, :])
np.save(f"{save_output_dir}/y_train.npy", y_arr[0:nr_train, :, :, :])
np.save(f"{save_output_dir}/x_test.npy", x_arr[nr_train:, :, :, :])
np.save(f"{save_output_dir}/y_test.npy", y_arr[nr_train:, :, :, :])