# -*- coding: utf-8 -*-
"""
Created on Mon May 29 15:46:44 2023

@author: NGPF
"""

# Import modules for data manipulation
import numpy as np
import os
import time
import datetime
import shutil
from tqdm import tqdm

# Import tensorflow
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import mixed_precision

# Enable FP16 mixed precision training
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)

# Enable multi-GPU computation
strategy = tf.distribute.MirroredStrategy(cross_device_ops=tf.distribute.HierarchicalCopyAllReduce())
print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

#%% Parameters

# Loss
def dice_coef_loss(y_true, y_pred, smooth=0):
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    dice = (2. * intersection + smooth) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth)
    return 1 - dice

def mse(y_true, y_pred):
    outputs = tf.square(y_true - y_pred)
    return tf.reduce_mean(outputs)

def total_loss(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    loss =  mse(y_true, y_pred)
    return tf.cast(loss, tf.float32)

def dice_coef_loss_per_sample(y_true, y_pred, smooth=0):
    num_sample = y_true.shape[0]
    loss_arr = None
    for i in range(0, num_sample):
        dice_loss = dice_coef_loss(y_true = tf.expand_dims(y_true[i,], axis=0), y_pred = tf.expand_dims(y_pred[i,], axis=0))
        if i == 0:
            loss_arr = dice_loss
        else:
            tf.stack([loss_arr, dice_loss], axis=0)
    if loss_arr == None:
        return tf.convert_to_tensor(np.ones((num_sample, 1), dtype='float32'))
    return loss_arr

def mse_per_sample(y_true, y_pred):
    outputs = tf.square(y_true - y_pred)
    return tf.reduce_mean(outputs, axis=list(range(1, len(y_true.shape))))

def loss_func_per_sample(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    loss = mse_per_sample(y_true, y_pred) #* 0.7 + dice_coef_loss_per_sample(y_true, y_pred) * 0.3
    return tf.cast(loss, tf.float32)

# Model saving
def create_save_model_dir(model_name, remark=""):
    time_of_training = str(datetime.datetime.now()).split(".")[0].replace(" ", "_").replace("-", "").replace(":", "")
    if remark != "":
        remark = f"_{remark}"
    training_name = f"{time_of_training}_{model_name}{remark}"
    save_base_dir = "H:/Kaggle Competitions/HuBMAP/hubmap-hacking-the-human-vasculature/trained_models"
    save_dir = f"{save_base_dir}/{training_name}"
    save_best_dir = f"{save_dir}/best_solution"
    save_final_dir = f"{save_dir}/final_solution"
    save_every_dir = f"{save_dir}/every_solution"
    os.mkdir(save_dir)
    os.mkdir(save_best_dir)
    os.mkdir(save_final_dir)
    os.mkdir(save_every_dir)
    print("Directory " + save_dir + " created." )
    return save_dir, save_best_dir, save_final_dir, save_every_dir

def delete_all_files_in_dir(directory):
    shutil.rmtree(directory)

def save_model(model_save_dir, model_to_save):
    if os.path.exists(model_save_dir):
        delete_all_files_in_dir(model_save_dir)
    else:
        os.mkdir(model_save_dir)
    model_to_save.save(model_save_dir)

def plot_KPI():
    print()

def decayed_learning_rate(initial_learning_rate, decay_rate, decay_steps, step):
    return initial_learning_rate * (decay_rate ** (step / decay_steps))

BATCH_SIZE = 4
GLOBAL_BATCH_SIZE = BATCH_SIZE * strategy.num_replicas_in_sync
EPOCH = 100

# Learning rate
INITIAL_LR = 0.001
# Mode
ENABLE_SC_LR_DECAY = True
ENABLE_CDT_LR_DECAY = True
# Scheduled
DECAY_STEP_SC = 1
DECAY_RATE_SC = 0.5
STAIRCASE = True
# Conditional
DECAY_RATE_CDT = 0.5

# Optimizer
with strategy.scope():
    if ENABLE_SC_LR_DECAY:
        lr_decay_schedule = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate = INITIAL_LR,
                                                                           decay_steps = DECAY_STEP_SC,
                                                                           decay_rate = DECAY_RATE_SC,
                                                                           staircase = STAIRCASE)
        OPTIMIZER = mixed_precision.LossScaleOptimizer(tf.keras.optimizers.SGD(learning_rate = lr_decay_schedule))
    else:
        OPTIMIZER = mixed_precision.LossScaleOptimizer(tf.keras.optimizers.SGD())

# Early stop
ES_MAX_EPOCH = 25
ES_MIN_VAR = 0.001

# Model saving
save_dir, save_best_dir, save_final_dir, save_every_dir = create_save_model_dir(model_name = "UNET_512_3", remark="LR_TEST")
SAVE_BEST = True
SAVE_EVERY = False
SAVE_FINAL = True

#%% Data Import

def dtype_cast(array, new_dtype):
    return array.astype(new_dtype)

# Import from NPY
base_dir = "H:/Kaggle Competitions/HuBMAP/hubmap-hacking-the-human-vasculature/train_prep"
x_train = np.load(f'{base_dir}/x_train.npy')
y_train = np.load(f'{base_dir}/y_train.npy')
x_test = np.load(f'{base_dir}/x_test.npy')
y_test = np.load(f'{base_dir}/y_test.npy')

# Cast to FP32
# x_train = dtype_cast(array=x_train, new_dtype='float32')
# y_train = dtype_cast(array=y_train, new_dtype='float32')
# x_test = dtype_cast(array=x_test, new_dtype='float32')
# y_test = dtype_cast(array=y_test, new_dtype='float32')

# Training set
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_dataset = train_dataset.shuffle(buffer_size=1024).batch(GLOBAL_BATCH_SIZE)
train_dataset = strategy.experimental_distribute_dataset(train_dataset)

# Validation set
val_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
val_dataset = val_dataset.batch(GLOBAL_BATCH_SIZE)
val_dataset = strategy.experimental_distribute_dataset(val_dataset)

#%% Model Definition

def custom_u_net_conv_block(x, unit=64, ker_size=3, stride=1, padding='same', batch_norm=True, repeat=2):
    for i in range(0, repeat):
        x = layers.Conv2D(unit, ker_size, stride, padding=padding)(x)
        x = layers.ReLU()(x)
        if batch_norm:
            x = layers.BatchNormalization()(x)
    return x


def custom_u_net_512_3():
    input = layers.Input(shape=(512,512,3))

    x = layers.Conv2D(64,3,1,padding='same')(input)
    x = layers.ReLU()(x)
    x = layers.BatchNormalization()(x)
    x = custom_u_net_conv_block(x, unit=64, repeat=1)
    a1 = x
    x = layers.MaxPooling2D(2,2)(a1)

    x = custom_u_net_conv_block(x, unit=128, repeat=2)
    a2 = x
    x = layers.MaxPooling2D(2,2)(a2)
    
    x = custom_u_net_conv_block(x, unit=256, repeat=2)
    a3 = x
    x = layers.MaxPooling2D(2,2)(a3)
    
    x = custom_u_net_conv_block(x, unit=512, repeat=2)
    a4 = x
    x = layers.MaxPooling2D(2,2)(a4)
    
    x = custom_u_net_conv_block(x, unit=1024, repeat=2)
    x = layers.BatchNormalization()(x)
    
    x = layers.Conv2DTranspose(128,1,2,padding='same')(x)
    
    x = layers.concatenate([x,a4],axis=-1)
    x = custom_u_net_conv_block(x, unit=512, repeat=2)
    
    x = layers.Conv2DTranspose(64,1,2,padding='same')(x)
    
    x = layers.concatenate([x,a3],axis=-1)
    x = custom_u_net_conv_block(x, unit=256, repeat=2)

    x = layers.Conv2DTranspose(32,1,2,padding='same')(x)
    
    x = layers.concatenate([x,a2],axis=-1)
    x = custom_u_net_conv_block(x, unit=128, repeat=2)

    x = layers.Conv2DTranspose(16,1,2,padding='same')(x)
    
    x = layers.concatenate([x,a1],axis=-1)
    x = custom_u_net_conv_block(x, unit=64, repeat=2)
    
    x = layers.Conv2D(1,1,1,padding='same', activation = 'sigmoid', dtype='float32')(x)
    
    model = tf.keras.Model(input,x)
    model.summary()
    
    return model

#%% Training Loop

with strategy.scope():
    def compute_loss(labels, predictions, model_losses):
        per_example_loss = loss_func_per_sample(labels, predictions)
        loss = tf.nn.compute_average_loss(per_example_loss, global_batch_size=GLOBAL_BATCH_SIZE)
        if model_losses:
          loss += tf.nn.scale_regularization_loss(tf.add_n(model_losses))
        return loss

    def compute_test_loss(labels, predictions):
        per_example_loss = loss_func_per_sample(labels, predictions)
        loss = tf.nn.compute_average_loss(per_example_loss, global_batch_size=GLOBAL_BATCH_SIZE)
        return loss

# Train
def train_step(inputs):
    images, labels = inputs
    with tf.GradientTape() as tape:
        predictions = model(images, training=True)
        loss = compute_loss(labels, predictions, model.losses)
        scaled_loss = OPTIMIZER.get_scaled_loss(loss)
    scaled_gradients = tape.gradient(scaled_loss, model.trainable_weights)
    gradients = OPTIMIZER.get_unscaled_gradients(scaled_gradients)
    OPTIMIZER.apply_gradients(zip(gradients, model.trainable_weights))
    return loss 

@tf.function
def distributed_train_step(dataset_inputs):
    per_replica_losses = strategy.run(train_step, args=(dataset_inputs,))
    return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)

# Test
def test_step(inputs):
    images, labels = inputs
    predictions = model(images, training=False)
    loss = compute_test_loss(labels, predictions)
    return loss

@tf.function
def distributed_test_step(dataset_inputs):
    per_replica_losses = strategy.run(test_step, args=(dataset_inputs,))
    return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)

with strategy.scope():
    model = custom_u_net_512_3()
hist_train_loss = []
hist_val_loss = []
last_n_val_loss = []
best_val_loss = 1.0

for epoch in range(0, EPOCH):
    print("\nStart of epoch %d" % (epoch,))
    start_time = time.time()

    # Train ##################################################################
    intermediate_train_loss = 0.0
    num_batches = 0

    for x in tqdm(train_dataset):
        intermediate_train_loss += distributed_train_step(x)
        num_batches += 1

    train_loss = intermediate_train_loss / num_batches
    print("Training loss: %.4f" % (float(train_loss),))

    # Test ###################################################################
    intermediate_test_loss = 0.0
    num_test_batches = 0

    for x in tqdm(val_dataset):
        intermediate_test_loss += distributed_test_step(x)
        num_test_batches += 1

    val_loss = intermediate_test_loss / num_test_batches

    print("Validation loss: %.4f" % (float(val_loss),))
    print("Time taken: %.2fs" % (time.time() - start_time))
    print("Current learning rate: %.7f" % float(decayed_learning_rate(initial_learning_rate = INITIAL_LR,
                                                                      decay_rate = DECAY_RATE_SC,
                                                                      decay_steps = DECAY_STEP_SC,
                                                                      step = epoch)))

    # Callbacks ##############################################################
    if last_n_val_loss != [] and val_loss + ES_MIN_VAR <= min(last_n_val_loss):
        best_val_loss = val_loss
        last_n_val_loss = []
        # Save best only
        save_model(model_save_dir = save_best_dir, model_to_save = model)
        print(f"Saving the best solution so far with validation loss = {best_val_loss}")
    last_n_val_loss.append(val_loss)

    # Earlystopping
    if len(last_n_val_loss) > ES_MAX_EPOCH:
        print(f"Validation loss does not improve over the last {ES_MAX_EPOCH} epochs, ending the training process...")
        break