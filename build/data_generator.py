import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator



import sys
sys.path.append('\VN-music-classification')  # Add parent directory to import varibles from config.py
from config import *

# train_dir = TRAIN_ROOT
# val_dir = VAL_ROOT
# test_dir = TEST_ROOT


def generate_data(train_dir, val_dir, test_dir, aug = False):
    """
    Generate train/val/test from datagen.flow_from_directory
    aug = False -> Create normal data. Otherwise, create real-time augmented data
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

    return train_generator, val_generator, test_generator



