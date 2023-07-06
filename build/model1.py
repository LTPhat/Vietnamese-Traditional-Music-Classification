import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
import os
import sys
sys.path.append('\VN-music-classification')  # Add parent directory to import varibles from config.py
from config import *
import os
import sys
sys.path.append('\VN-music-classification')  # Add parent directory to import varibles from config.py
from config import *




def get_model1(input_shape = INPUT_SHAPE, n_class = N_CLASS):
    model1 = tf.keras.models.Sequential([
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
    filepath = CHECKPOINT_FILEPATH + '\\model1' + '\\model1_{epoch:02d}_{val_accuracy:.4f}.h5',
    monitor = CHECKPOINT_MONITOR,
    save_best_only = True,
    save_weights_only = True,
    verbose = 1
    )

    # Define callbacks
    early = EarlyStopping(monitor = EARLY_MONITOR,
        patience = PATIENCE,
        verbose = VERBOSE,
        mode='auto',
        baseline= None,
        restore_best_weights= True)
    
    return model1, checkpoint1, early

    