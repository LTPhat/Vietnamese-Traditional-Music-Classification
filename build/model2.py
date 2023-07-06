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



def get_model2(input_shape = INPUT_SHAPE, n_class = N_CLASS):

    model2= tf.keras.Sequential(layers=[
            tf.keras.layers.InputLayer(input_shape= (input_shape[0], input_shape[1], 3)),
            # first convolution
            tf.keras.layers.Conv2D(32, (5, 5), padding='same', activation="relu"),
            tf.keras.layers.MaxPooling2D(),
            # second convolution
            tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation="relu"),
            tf.keras.layers.MaxPooling2D(),
            # third convolution
            tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation="relu"),
            tf.keras.layers.MaxPooling2D(),
            # FC 
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(n_class, activation="softmax")
        ])
    
    # Create dir to save checkpoint
    if not os.path.exists(CHECKPOINT_FILEPATH + '\\model2'):
        os.makedirs(CHECKPOINT_FILEPATH + '\\model2')

    # Define checkpoint
    checkpoint2= tf.keras.callbacks.ModelCheckpoint(
    filepath = CHECKPOINT_FILEPATH + '\\model2' + '\\model2_{epoch:02d}_{val_accuracy:.4f}.h5',
    monitor = CHECKPOINT_MONITOR,
    save_best_only=True,
    save_weights_only=True,
    verbose=1
    )

    # Define callbacks
    early = EarlyStopping(monitor = EARLY_MONITOR,
        patience = PATIENCE,
        verbose = VERBOSE,
        mode='auto',
        baseline= None,
        restore_best_weights= True)

    return model2, checkpoint2, early
    