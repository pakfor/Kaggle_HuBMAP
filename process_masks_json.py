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
OUTPUT_PNG = "H:/Kaggle Competitions/HuBMAP/hubmap-hacking-the-human-vasculature/train_mask/Blood Vessel/PNG"
OUTPUT_NPY = "H:/Kaggle Competitions/HuBMAP/hubmap-hacking-the-human-vasculature/train_mask/Blood Vessel/NPY"
mask_txt_dir = "H:/Kaggle Competitions/HuBMAP/hubmap-hacking-the-human-vasculature/polygons.jsonl"

with open(mask_txt_dir) as f:
    data = f.readlines()
    f.close()

for mask in tqdm(data):
    mask_map = np.zeros((MAP_SIZE, MAP_SIZE, 1))

    mask_dict = eval(mask)
    mask_id = mask_dict["id"]

    for each_annotation in mask_dict["annotations"]:
        annotation_type = each_annotation["type"]
        coordinates_list = each_annotation["coordinates"]

        if annotation_type == 'blood_vessel':
            mask_map = cv2.fillPoly(mask_map, [np.array(coordinates_list)], (1,1,1))

    plt.imsave(f"{OUTPUT_PNG}/{mask_id}.png", np.reshape(mask_map, (MAP_SIZE, MAP_SIZE)), cmap='gray')
    np.save(f"{OUTPUT_NPY}/{mask_id}.npy", mask_map.astype('float16'))
    #plt.imshow(mask_map, cmap='gray')
    #plt.show()
