import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from sklearn.metrics import confusion_matrix
import seaborn as sns
from keras.preprocessing import image
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from data_generator import generate_data
import os
import sys
sys.path.append('\VN-music-classification')  # Add parent directory to import varibles from config.py
from config import *



train_dir = DATASET_ROOT + "\\" + "train"
val_dir = DATASET_ROOT + "\\" + "val"
test_dir = DATASET_ROOT + "\\" + "test"



def model1(input_shape = INPUT_SHAPE, n_class = N_CLASS):
    model = tf.keras.models.Sequential([
        #first_convolution
        tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(input_shape[0], input_shape[1], 3)),
        tf.keras.layers.MaxPooling2D(2, 2),
        #second_convolution
        tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        #third_convolution
        tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        #fourth_convolution
        tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(n_class, activation='softmax')
    ])

    if not os.path.exists(CHECKPOINT_FILEPATH + '\\model1'):
        os.makedirs(CHECKPOINT_FILEPATH + '\\model1')

    # Define checkpoint
    checkpoint1= tf.keras.callbacks.ModelCheckpoint(
    filepath= CHECKPOINT_FILEPATH + '\\model1' + '\\model1_{epoch:02d}_{val_accuracy:.4f}.h5',
    monitor='val_accuracy',
    save_best_only=True,
    save_weights_only=True,
    verbose=1
    )

    # Define callbacks
    early = EarlyStopping(monitor='loss',
        patience= 5,
        verbose= 1,
        mode='auto',
        baseline= None,
        restore_best_weights= True)
    
    return model, checkpoint1, early


if __name__ == '__main__':
    train_model1()