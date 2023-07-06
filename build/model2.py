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



def model2(input_shape = INPUT_SHAPE, n_class = N_CLASS):

    model2 = tf.keras.Sequential(layers=[
            tf.keras.layers.InputLayer(input_shape= (input_shape[0], input_shape[1], 3)),
            tf.keras.layers.Conv2D(32, (5, 5), padding='same', activation="relu"),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation="relu"),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation="relu"),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(n_class, activation="softmax")
        ])
    return model2


def train_model2(model):
    # Define callbacks
    early = EarlyStopping(monitor='loss',
        patience= 5,
        verbose= 1,
        mode='auto',
        baseline= None,
        restore_best_weights= True)
    
    # Define checkpoint
    if not os.path.exists(CHECKPOINT_FILEPATH + '/model2'):
        os.makedirs(CHECKPOINT_FILEPATH + '/model2')
        
    checkpoint2 = tf.keras.callbacks.ModelCheckpoint(
    filepath= CHECKPOINT_FILEPATH + '/model2' + '/model2_{epoch:02d}_{val_accuracy:.4f}.h5',
    monitor='val_accuracy',
    save_best_only=True,
    save_weights_only=True,
    verbose=1
    )

    
    # Get non-augmented data to train
    non_train_generator, non_val_generator, non_test_generator = generate_data(train_dir = train_dir, val_dir = val_dir, test_dir = test_dir)

    # Get augmented data to train (If needed)
    # train_generator, val_generator, _test_generator = generate_data(train_dir = train_dir, val_dir = val_dir, test_dir = test_dir, aug = True)


    # Fit with non-augmented data
    non_model1_history = non_model2.fit(non_train_generator, batch_size= 32, epochs = 20, callbacks=[early, checkpoint1],
                   validation_data = non_val_generator, validation_batch_size = 32)
    return non_model1_history

if __name__ == '__main__':
    train_model2()