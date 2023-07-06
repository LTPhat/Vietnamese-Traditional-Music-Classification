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

import sys
sys.path.append('\VN-music-classification')  # Add parent directory to import varibles from config.py
from config import *


# Define train/test/val directory

train_dir = DATASET_ROOT + "\\" + "train"
val_dir = DATASET_ROOT + "\\" + "val"
test_dir = DATASET_ROOT + "\\" + "test"



def generate_data(train_dir, val_dir, test_dir, aug = False):
    """
    Generate train/val/test from datagen.flow_from_directory
    Return 
    """
    if not aug:
        train_datagen = ImageDataGenerator(rescale = RESCALE)
        val_datagen = ImageDataGenerator(rescale = RESCALE)
        test_datagen = ImageDataGenerator(rescale = RESCALE)


        train_generator = train_datagen.flow_from_directory(
            train_dir,
            target_size= INPUT_SHAPE,
            shuffle = True,
            subset='training'
        )
        val_generator = val_datagen.flow_from_directory(
            val_dir,
            target_size= INPUT_SHAPE,
        )
        test_generator = test_datagen.flow_from_directory(
            test_dir,
            target_size= INPUT_SHAPE,
        )
    else: 
                # Create dataset with data augmentation
        train_datagen = ImageDataGenerator(rescale = RESCALE,
                                        width_shift_range = WIDTH_SHIFT_RANGE, height_shift_range = HEIGHT_SHIFT_RANGE,
                                            zoom_range = ZOOM_RANGE)
        val_datagen = ImageDataGenerator(rescale = RESCALE)
        test_datagen = ImageDataGenerator(rescale = RESCALE)

        train_generator = train_datagen.flow_from_directory(
            train_dir,
            target_size = INPUT_SHAPE,
            shuffle = True,
            subset='training'
        )
        val_generator = val_datagen.flow_from_directory(
            val_dir,
            target_size= INPUT_SHAPE,
        )
        test_generator = test_datagen.flow_from_directory(
            test_dir,
            target_size= INPUT_SHAPE,
        )

    return train_generator, val_generator, test_datagen



