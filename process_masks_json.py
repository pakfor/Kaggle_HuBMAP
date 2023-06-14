# -*- coding: utf-8 -*-
"""
Created on Fri May 26 18:51:14 2023

@author: NGPF
"""
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import cv2

MAP_SIZE = 512
OUTPUT_PNG_glomerulus = "H:/Kaggle Competitions/HuBMAP/hubmap-hacking-the-human-vasculature/train_mask/glomerulus/PNG"
OUTPUT_NPY_glomerulus  = "H:/Kaggle Competitions/HuBMAP/hubmap-hacking-the-human-vasculature/train_mask/glomerulus/NPY"
OUTPUT_PNG_unsure = "H:/Kaggle Competitions/HuBMAP/hubmap-hacking-the-human-vasculature/train_mask/unsure/PNG"
OUTPUT_NPY_unsure = "H:/Kaggle Competitions/HuBMAP/hubmap-hacking-the-human-vasculature/train_mask/unsure/NPY"
mask_txt_dir = "H:/Kaggle Competitions/HuBMAP/hubmap-hacking-the-human-vasculature/polygons.jsonl"

with open(mask_txt_dir) as f:
    data = f.readlines()
    f.close()

for mask in tqdm(data):
    mask_map_unsure = np.zeros((MAP_SIZE, MAP_SIZE, 1))
    mask_map_glomerulus = np.zeros((MAP_SIZE, MAP_SIZE, 1))

    mask_dict = eval(mask)
    mask_id = mask_dict["id"]

    for each_annotation in mask_dict["annotations"]:
        annotation_type = each_annotation["type"]
        coordinates_list = each_annotation["coordinates"]

        if annotation_type == 'glomerulus':
            mask_map_glomerulus = cv2.fillPoly(mask_map_glomerulus, [np.array(coordinates_list)], (1,1,1))
        elif annotation_type == 'unsure':
            mask_map_unsure = cv2.fillPoly(mask_map_unsure, [np.array(coordinates_list)], (1,1,1))
        else:
            print()

    plt.imsave(f"{OUTPUT_PNG_glomerulus}/{mask_id}.png", np.reshape(mask_map_glomerulus, (MAP_SIZE, MAP_SIZE)), cmap='gray')
    np.save(f"{OUTPUT_NPY_glomerulus}/{mask_id}.npy", mask_map_glomerulus.astype('float16'))
    plt.imsave(f"{OUTPUT_PNG_unsure}/{mask_id}.png", np.reshape(mask_map_unsure, (MAP_SIZE, MAP_SIZE)), cmap='gray')
    np.save(f"{OUTPUT_NPY_unsure}/{mask_id}.npy", mask_map_unsure.astype('float16'))
    #plt.imshow(mask_map, cmap='gray')
    #plt.show()
