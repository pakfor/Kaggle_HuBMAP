# -*- coding: utf-8 -*-
"""
Created on Sat May 27 17:21:25 2023

@author: NGPF
"""
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

# Metrics
# DICE
def dice(pred, true, k = 1):
    intersection = np.sum(pred[true==k]) * 2.0
    dice = intersection / (np.sum(pred) + np.sum(true))
    return dice

# MSE
def mse(pred, true):
    return np.square(pred - true).mean()

# ROC & Confusion matrix
def clean_acc_threshold(array, threshold):
    array[array >= threshold] = 1
    array[array < threshold] = 0
    return array

def confusion_matrix_component_cal(pred, true, threshold):
    # True Positive (TP): Diagnosed with positive, ground truth is positive
    # False Positive (FP): Diagnosed with positive, ground truth is negative
    # True Negative (TN): Diagnosed with negative, ground truth is negative
    # False Negative (FN): Diagnosed with negative, ground truth is positive
    pred_clean = clean_acc_threshold(array = pred, threshold = threshold)
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    for i in range(0, 512):
        for j in range(0, 512):
            pred_pix = pred_clean[i, j]
            gt_pix = true[i, j]
            if pred_pix == gt_pix:
                # pred = 1, true = 1
                if pred_pix == 1.0:
                    TP += 1
                # pred = 0, true = 0
                else:
                    TN += 1
            else:
                # pred = 1, true = 0
                if pred_pix == 1.0:
                    FP += 1
                # pred = 0, true = 1
                else:
                    FN += 1
    return TP, FP, TN, FN

def plot_ROC(thresholds, roc_components_matrix):
    tpr_avg = []
    fpr_avg = []
    for t in range(0, len(thresholds)):
        tpr_avg.append(np.average(roc_components_matrix[:, t, 4]))
        fpr_avg.append(np.average(roc_components_matrix[:, t, 5]))
    plt.plot(fpr_avg, tpr_avg, color='green', linestyle='dashed', marker='o')
    plt.show()

def triple_plot(img_to_predict, prediction, ground_truth):
    fig, ax = plt.subplots(1, 3)
    ax[0].imshow(img_to_predict)
    ax[0].axis('off')
    ax[1].imshow(prediction, cmap='gray', vmin=0, vmax=1)
    ax[1].axis('off')
    ax[2].imshow(ground_truth, cmap='gray', vmin=0, vmax=1)
    ax[2].axis('off')
    plt.show()

thresholds = [0.01, 0.05, 0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99]

trained_model_dir = "H:/Kaggle Competitions/HuBMAP/hubmap-hacking-the-human-vasculature/trained_models/20230610_202133_UNET_512_3/best_solution"
save_dir_PNG = "H:/Kaggle Competitions/HuBMAP/hubmap-hacking-the-human-vasculature/trained_models/20230527_UNet_T1/predictions/PNG"
save_dir_NPY = "H:/Kaggle Competitions/HuBMAP/hubmap-hacking-the-human-vasculature/trained_models/20230527_UNet_T1/predictions/NPY"
x_test = np.load("H:/Kaggle Competitions/HuBMAP/hubmap-hacking-the-human-vasculature/train_prep/x_test.npy")
y_test = np.load("H:/Kaggle Competitions/HuBMAP/hubmap-hacking-the-human-vasculature/train_prep/y_test.npy")
stain_norm_dir = "H:/Kaggle Competitions/HuBMAP/hubmap-hacking-the-human-vasculature/train_stain_norm"
stain_norm_test_list = os.listdir(stain_norm_dir)[1000:]

trained_model = tf.keras.models.load_model(trained_model_dir)

results_dict = {"Input":[], "DICE Loss":[], "MSE Loss":[]}

# roc_components_matrix[sample, threshold, TP/FP/TN/FN/TPR/FPR]
# TPR = TP / (TP + FN)
# FPR = FP / (FP + TN)
roc_components_matrix = np.zeros((x_test.shape[0], len(thresholds), 6))

for i in tqdm(range(0, x_test.shape[0])):
    img_to_predict = x_test[i, :, :, :].astype('float32')

    prediction = trained_model.predict(np.reshape(img_to_predict, (1, 512, 512, 3)).astype('float16'), verbose=0)
    prediction = np.reshape(prediction, (512, 512)).astype('float32')

    ground_truth = y_test[i, :, :, :]
    ground_truth = np.reshape(ground_truth, (512, 512)).astype('float32')

    # triple_plot(img_to_predict = img_to_predict,
    #             prediction = prediction,
    #             ground_truth = ground_truth)

    # Calculate metrics, ensuring the data are in float32 format
    dice_loss = dice(pred = prediction, true = ground_truth)
    mse_loss = mse(pred = prediction, true = ground_truth)

    # ROC & Confusion Matrix
    for t in range(0, len(thresholds)):
        threshold_value = thresholds[t]
        tp, fp, tn, fn = confusion_matrix_component_cal(pred = prediction,
                                                        true = ground_truth,
                                                        threshold = threshold_value)
        if tp + fn == 0:
            tpr = 0
        else:
            tpr = tp / (tp + fn)
        
        if fp + tn == 0:
            fpr = 0
        else:
            fpr = fp / (fp + tn)

        roc_components_matrix[i, t, 0] = tp
        roc_components_matrix[i, t, 1] = fp
        roc_components_matrix[i, t, 2] = tn
        roc_components_matrix[i, t, 3] = fn
        roc_components_matrix[i, t, 4] = tpr
        roc_components_matrix[i, t, 5] = fpr

plot_ROC(thresholds = thresholds, roc_components_matrix = roc_components_matrix)

    # plt.imshow(img_to_predict)
    # plt.show()
    # plt.imshow(prediction, cmap='gray', vmin=0, vmax=1)
    # plt.show()
    # plt.imshow(ground_truth, cmap='gray', vmin=0, vmax=1)
    # plt.show()