#from utils import *

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pathlib
import csv
import random

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.layers.core import Dense, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam


# set random seed
rdm = 1
random.seed(rdm)
np.random.seed(rdm)
tf.random.set_seed(rdm)


# obtain dataset directory filepath
current_dir = pathlib.Path.cwd()
movi_dir = current_dir.parent.parent
movi2img_dir = movi_dir / ('movi2img')


# parameters
batch_size = 8
img_height = 255
img_width = 255

layer1_list = [40, 48, 56, 64]
layer2 = 20

epochs = 30
learning_rate_list = [0.001, 0.005]


for i in range(len(learning_rate_list)):
    for j in range(len(layer1_list)):
        learning_rate = learning_rate_list[i]
        layer1 = layer1_list[j]
        
        # load ResNet50 and add custom layers
        resnet_model = Sequential()

        pretrained_model= tf.keras.applications.ResNet50(
            include_top = False, # ensures that we can add our own custom input and output layers
            input_shape = (255, 255, 3), # shape of each input image (ndarray)
            pooling = 'avg', # feature extraction
            weights = 'imagenet') # Resnet50 model will use the weights it learnt while being trained on the imagenet data
        
        # create neural network
        resnet_model.add(pretrained_model)
        resnet_model.add(Flatten())
        resnet_model.add(Dense(layer1, activation='relu'))
        resnet_model.add(Dense(layer2, activation='softmax'))
        resnet_model.summary()

        # obtain training and validation datasets
        train_ds = tf.keras.utils.image_dataset_from_directory(
            movi2img_dir,
            labels='inferred',
            label_mode='categorical',
            validation_split=0.2,
            subset="training",
            seed=1,
            image_size=(img_height, img_width),
            batch_size=batch_size)

        val_ds = tf.keras.utils.image_dataset_from_directory(
            movi2img_dir,
            labels='inferred',
            label_mode='categorical',
            validation_split=0.2,
            subset="validation",
            seed=1,
            image_size=(img_height, img_width),
            batch_size=batch_size)


        # compile and fit model
        resnet_model.compile(optimizer=Adam(learning_rate=learning_rate),loss='categorical_crossentropy',metrics=['accuracy'])
        history = resnet_model.fit(train_ds, validation_data=val_ds, epochs=epochs)


        # plot accuracy graph
        fig1 = plt.gcf()
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.axis(ymin=0.4,ymax=1)
        plt.grid()
        plt.title('Model Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epochs')
        plt.legend(['train', 'validation'], loc='upper left')

        # save accuracy graph as image file
        lr_sci_notation = "{:.0E}".format(learning_rate)
        accuracy_fname = f'accuracy_{layer1}-{layer2}_b{batch_size}_e{epochs}_lr{lr_sci_notation}.png'
        plt.savefig(accuracy_fname)
        plt.close()

        # plot loss and save to file
        fig2 = plt.gcf()
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epochs')
        plt.legend(['train', 'validation'], loc='upper left')

        # save loss graph as image file
        loss_fname = f'loss_{layer1}-{layer2}_b{batch_size}_e{epochs}_lr{lr_sci_notation}.png'
        plt.savefig(loss_fname)
        plt.close()


        # save model weights
        model_fname = f'model_{layer1}-{layer2}_b{batch_size}_e{epochs}_lr{lr_sci_notation}.h5'
        resnet_model.save(model_fname)
