# -*- coding: utf-8 -*-

from __future__ import division
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import models_small as M
import numpy as np
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
import pickle
import argparse
import glob
from liver_seg_generator25 import DataGenerator
import matplotlib.pyplot as plt
import Reza_functions2 as rf

parser = argparse.ArgumentParser()
parser.add_argument('-e',type=int)
parser.add_argument('-p', type=str)
parser.add_argument('-w', type=str)
args = parser.parse_args()


# Parameters
print("set Parameters...")
params = {'dim': (256,256),
          'batch_size': 16,
          'n_classes': 1, # 這個好像要處理一下，有看到說可以用1，或4的倍數
          'n_channels': 3,
          'shuffle': True
          }

nb_epoch = args.e
print('Get Dataset Path')

folder = 'liver_npy/'
lung_window_Paths = glob.glob(folder+"Dicom_lung*")
soft_window_Paths = glob.glob(folder+"Dicom_soft*")
bone_window_Paths = glob.glob(folder+"Dicom_bone*")
Mask_Paths=glob.glob(folder+"MaskA_*")


lung_window_Paths.sort(key = rf.natural_keys)
soft_window_Paths.sort(key = rf.natural_keys)
bone_window_Paths.sort(key = rf.natural_keys)
Mask_Paths.sort(key = rf.natural_keys)



splitIdx = int(len(lung_window_Paths)*0.9)


# Generators
training_generator = DataGenerator(lung_window_Paths[:splitIdx],
                                    soft_window_Paths[:splitIdx],
                                    bone_window_Paths[:splitIdx],
                                    Mask_Paths[:splitIdx],
                                    **params)

validation_generator = DataGenerator(lung_window_Paths[splitIdx:],
                                    soft_window_Paths[splitIdx:],
                                    bone_window_Paths[splitIdx:],
                                    Mask_Paths[splitIdx:],
                                    **params)

#################################### Build Model #####################################
# Build model
model = M.unet(input_size = (256,256,3))
model.summary()
if type(args.p)=='str':
    model.load_weights(args.p) #pretrain


print('Training')
weight=args.w
# 由於訓練過程耗時，有可能訓練一半就當掉，因此，我們可以利用這個 Callback，在每一個檢查點(Checkpoint)存檔，下次執行時，就可以從中斷點繼續訓練。
mcp_save = ModelCheckpoint(weight, save_best_only=True, monitor='val_loss', mode='min')
# 當訓練已無改善時，可以降低學習率，追求更細微的改善，找到更精準的最佳解。
reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=7, verbose=1, epsilon=1e-4, mode='min')



# Train model on dataset

history = model.fit_generator(generator=training_generator,
                    validation_data=validation_generator,
                    epochs=nb_epoch,
                    workers=6,
                    use_multiprocessing=True,
                    verbose=1,
                    callbacks=[mcp_save, reduce_lr_loss]) 

print('Trained model saved')
with open('hist_liver', 'wb') as file_pi:
        pickle.dump(history.history, file_pi)
print('Trained model saved')
with open('hist_liver', 'wb') as file_pi:
        pickle.dump(history.history, file_pi)

rf.history_plot(history,weight)




