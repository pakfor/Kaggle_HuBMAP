# -*- coding: utf-8 -*-
"""
Created on Sat May 27 17:21:25 2023

@author: NGPF
"""
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

trained_model_dir = "H:/Kaggle Competitions/HuBMAP/hubmap-hacking-the-human-vasculature/trained_models/20230527_UNet_T1/model_ckpt"
save_dir_PNG = "H:/Kaggle Competitions/HuBMAP/hubmap-hacking-the-human-vasculature/trained_models/20230527_UNet_T1/predictions/PNG"
save_dir_NPY = "H:/Kaggle Competitions/HuBMAP/hubmap-hacking-the-human-vasculature/trained_models/20230527_UNet_T1/predictions/NPY"
x_test = np.load("H:/Kaggle Competitions/HuBMAP/hubmap-hacking-the-human-vasculature/train_prep/x_test.npy")
y_test = np.load("H:/Kaggle Competitions/HuBMAP/hubmap-hacking-the-human-vasculature/train_prep/y_test.npy")
stain_norm_dir = "H:/Kaggle Competitions/HuBMAP/hubmap-hacking-the-human-vasculature/train_stain_norm"
stain_norm_test_list = os.listdir(stain_norm_dir)[1000:]

trained_model = tf.keras.models.load_model(trained_model_dir)

for i in range(0, x_test.shape[0]):
    img_to_predict = x_test[i, :, :, :].astype('float32')

    prediction = trained_model.predict(np.reshape(img_to_predict, (1, 512, 512, 3)).astype('float16'))
    prediction = np.reshape(prediction, (512, 512)).astype('float32')

    ground_truth = y_test[i, :, :, :]
    ground_truth = np.reshape(ground_truth, (512, 512)).astype('float32')

    fig, ax = plt.subplots(1, 3)
    ax[0].imshow(img_to_predict)
    ax[0].axis('off')
    ax[1].imshow(prediction, cmap='gray', vmin=0, vmax=1)
    ax[1].axis('off')
    ax[2].imshow(ground_truth, cmap='gray', vmin=0, vmax=1)
    ax[2].axis('off')
    plt.show()

    # plt.imshow(img_to_predict)
    # plt.show()
    # plt.imshow(prediction, cmap='gray', vmin=0, vmax=1)
    # plt.show()
    # plt.imshow(ground_truth, cmap='gray', vmin=0, vmax=1)
    # plt.show()