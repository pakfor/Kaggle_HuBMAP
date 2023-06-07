# -*- coding: utf-8 -*-
"""
Created on Sat May 27 08:50:34 2023

@author: NGPF
"""

# Import modules for data manipulation
import numpy as np
import os
import time 

# Import tensorflow
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import mixed_precision

# Enable FP16 mixed precision training
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)

# Enable multi-GPU computation
mirrored_strategy = strategy = tf.distribute.MirroredStrategy(cross_device_ops = tf.distribute.ReductionToOneDevice())
devices = tf.config.experimental.list_physical_devices("GPU")

#%% Data Import

base_dir = "H:/Kaggle Competitions/HuBMAP/hubmap-hacking-the-human-vasculature/train_prep"
x_train = np.load(f'{base_dir}/x_train.npy')
y_train = np.load(f'{base_dir}/y_train.npy')
x_test = np.load(f'{base_dir}/x_test.npy')
y_test = np.load(f'{base_dir}/y_test.npy')

# =============================================================================
# # Make Sure All Data are Float16
# x_train = x_train.astype('float16')
# y_train = y_train.astype('float16')
# x_test = x_test.astype('float16')
# y_test = y_test.astype('float16')
# 
# # Shuffle the data
# shuffler = np.random.permutation(len(x_train))
# x_train = x_train[shuffler]
# y_train = y_train[shuffler]
# shuffler = np.random.permutation(len(x_test))
# x_test = x_test[shuffler]
# y_test = y_test[shuffler]
# =============================================================================

#%% Data Generator

batch_size = 16

# Define generator for training data according to batch size as defined by user
def train_gen(x_train,y_train):
    shuffler = np.random.permutation(len(x_train))
    x_train = x_train[shuffler]
    y_train = y_train[shuffler]
    
    while True:
        for i in range(0,len(x_train),batch_size):
            yield (x_train[i:i+batch_size],y_train[i:i+batch_size])

# Define generator for testing data, batch size is set to 1
def validate_gen(x_test,y_test):
    shuffler = np.random.permutation(len(x_test))
    x_test = x_test[shuffler]
    y_test = y_test[shuffler]
    
    while True:
        for i in range(0,len(x_test),1):
            yield (x_test[i:i+1],y_test[i:i+1])

#%% Loss

def dice_coef(y_true, y_pred, smooth=0):
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    dice = (2. * intersection + smooth) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth)
    return dice

def dice_coef_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)

def mse(y_true, y_pred):
    error = tf.subtract(y_true, y_pred)
    sqr_error = tf.square(error)
    mean_sqr_error = tf.reduce_mean(sqr_error)
    return mean_sqr_error

def mse_old(y_true, y_pred):
    error = y_true - y_pred
    sqr_error = tf.keras.backend.square(error)
    mean_sqr_error = tf.keras.backend.mean(sqr_error)
    return mean_sqr_error

def total_loss(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    loss =  mse(y_true, y_pred) # + dice_coef_loss(y_true, y_pred)
    return tf.cast(loss, tf.float32)

#%% Neural Network

# Define the architecture U-Net of the neural network to be trained for segmentation
def UNet():
    input = layers.Input(shape=(512,512,3))
    x = layers.Conv2D(64,3,1,padding='same')(input)
    x = layers.ReLU()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(64,3,1,padding='same')(x)
    x = layers.ReLU()(x)
    x = layers.BatchNormalization()(x)
    a1 = x
    
    x = layers.MaxPooling2D(2,2)(a1)
    
    x = layers.Conv2D(128,3,1,padding='same')(x)
    x = layers.ReLU()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(128,3,1,padding='same')(x)
    x = layers.ReLU()(x)
    x = layers.BatchNormalization()(x)
    a2 = x
    
    x = layers.MaxPooling2D(2,2)(a2)
    
    x = layers.Conv2D(256,3,1,padding='same')(x)
    x = layers.ReLU()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(256,3,1,padding='same')(x)
    x = layers.ReLU()(x)
    x = layers.BatchNormalization()(x)
    a3 = x
    
    x = layers.MaxPooling2D(2,2)(a3)
    
    x = layers.Conv2D(512,3,1,padding='same')(x)
    x = layers.ReLU()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(512,3,1,padding='same')(x)
    x = layers.ReLU()(x)
    x = layers.BatchNormalization()(x)
    a4 = x
    
    x = layers.MaxPooling2D(2,2)(a4)
    
    x = layers.Conv2D(1024,3,1,padding='same')(x)
    x = layers.ReLU()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(1024,3,1,padding='same')(x)
    x = layers.ReLU()(x)
    x = layers.BatchNormalization()(x)
    
    x = layers.Conv2DTranspose(128,1,2,padding='same')(x)
    
    x = layers.concatenate([x,a4],axis=-1)
    x = layers.Conv2D(512,3,1,padding='same')(x)
    x = layers.ReLU()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(512,3,1,padding='same')(x)
    x = layers.ReLU()(x)
    x = layers.BatchNormalization()(x)
    
    x = layers.Conv2DTranspose(64,1,2,padding='same')(x)
    
    x = layers.concatenate([x,a3],axis=-1)
    x = layers.Conv2D(256,3,1,padding='same')(x)
    x = layers.ReLU()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(256,3,1,padding='same')(x)
    x = layers.ReLU()(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv2DTranspose(32,1,2,padding='same')(x)
    
    x = layers.concatenate([x,a2],axis=-1)
    x = layers.Conv2D(128,3,1,padding='same')(x)
    x = layers.ReLU()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(128,3,1,padding='same')(x)
    x = layers.ReLU()(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv2DTranspose(16,1,2,padding='same')(x)
    
    x = layers.concatenate([x,a1],axis=-1)
    x = layers.Conv2D(64,3,1,padding='same')(x)
    x = layers.ReLU()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(64,3,1,padding='same')(x)
    x = layers.ReLU()(x)
    x = layers.BatchNormalization()(x)
    
    x = layers.Conv2D(1,1,1,padding='same', activation = 'sigmoid', dtype='float32')(x)
    
    model = tf.keras.Model(input,x)
    model.summary()
    
    return model

#%% Model Compile & Training

# Start the timer for recording the time taken for training the model
train_start = time.time()

# Create saving paths
save_dir = 'H:/Kaggle Competitions/HuBMAP/hubmap-hacking-the-human-vasculature/trained_models/20230529_UNet_TEST_CUSTOM_LOSS'
if not os.path.exists(save_dir):
    os.mkdir(save_dir)

model_save = f'{save_dir}/model' # For saving the result at the end of the training
log_save = f'{save_dir}/log' # For storing training activities
ckpt_save = f'{save_dir}/model_ckpt' # For saving the best weightings recorded during the training (continuously updated during training)

# Create the directories as mentioned above
if not os.path.exists(model_save):
    os.mkdir(model_save)
if not os.path.exists(log_save):
    os.mkdir(log_save)
if not os.path.exists(ckpt_save):
    os.mkdir(ckpt_save)

# Define optimizer: Adam
opt_adam = tf.keras.optimizers.Adam(clipvalue=1.0)

# Define callbacks: early stopping and model checkpoint
earlystop = tf.keras.callbacks.EarlyStopping(monitor = 'val_loss', min_delta = 0.001, patience = 20, verbose = 1, mode = 'min')
model_ckpt = tf.keras.callbacks.ModelCheckpoint(ckpt_save, monitor = 'val_loss', verbose = 1, save_best_only = True, mode = 'min')

with mirrored_strategy.scope():
    # Make model
    model = UNet()
    # Compile the model by defining loss function (MSE) and optimizer (Adam)
    model.compile('Adam',
                  loss = total_loss,
                  metrics = ['accuracy'])

# Fit the model with training data and validate it with testing data every epoch
# Define maximum number of epochs
# Apply callbacks
model.fit(train_gen(x_train, y_train), steps_per_epoch = len(x_train) / batch_size, epochs = 200,
          validation_data = validate_gen(x_test, y_test), validation_steps = len(x_test),
          callbacks = [model_ckpt, earlystop])

# Stop the timer for recording the time taken for training the model
train_end = time.time()
# Calculate time taken and print it
elapse = train_end - train_start
print(elapse)

# Save the trained model to the designated directory
model.save(model_save)