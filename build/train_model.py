import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
import os
import sys
sys.path.append('\VN-music-classification')  # Add parent directory to import varibles from config.py
from data_generator import generate_data
from config import *
from model1 import get_model1
from model2 import get_model2
from model3 import get_model3

# Define model
model1, checkpoint1, early1 = get_model1(INPUT_SHAPE, N_CLASS)
model2, checkpoint2, early2 = get_model2(INPUT_SHAPE, N_CLASS)
model3, checkpoint3, early3 = get_model3(INPUT_SHAPE, N_CLASS)


def train_model(model, checkpoint, early, train_set, val_set, test_set):
    model.compile(loss = LOSS, optimizer = OPTIMIZER, metrics= METRICS)
    model_history = model.fit(train_set, batch_size= BATCH_SIZE, epochs = EPOCHS, callbacks=[early, checkpoint],
                   validation_data = val_set, validation_batch_size = VALIDATION_BATCH_SIZE)
    return model_history


def load_best_model(model_index, val_set):
    """
    Load best model after training
    model_index = {1, 2, 3} - Load best model with highest val_acc of model1, 2, 3
    """
    if model_index == 1:
        best_model, _ , _ = get_model1(input_shape = INPUT_SHAPE, n_class = N_CLASS)
        checkpoint_model_path = CHECKPOINT_FILEPATH + "/model1"
        best_model_path =  checkpoint_model_path + "/" + str(os.listdir(checkpoint_model_path)[-1])
        best_model.load_weights(best_model_path)
        best_model.compile(loss = LOSS, optimizer = OPTIMIZER, metrics=[METRICS[0], tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])
    elif model_index == 2:
        best_model, _ , _ = get_model2(input_shape = INPUT_SHAPE, n_class = N_CLASS)
        checkpoint_model_path = CHECKPOINT_FILEPATH + "/model2"
        best_model_path =  checkpoint_model_path + "/" + str(os.listdir(checkpoint_model_path)[-1])
        best_model.load_weights(best_model_path)
        best_model.compile(loss = LOSS, optimizer = OPTIMIZER, metrics=[METRICS[0], tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])
    elif model_index == 3:
        best_model, _ , _ = get_model2(input_shape = INPUT_SHAPE, n_class = N_CLASS)
        checkpoint_model_path = CHECKPOINT_FILEPATH + "/model3"
        best_model_path =  checkpoint_model_path + "/" + str(os.listdir(checkpoint_model_path)[-1])
        best_model.load_weights(best_model_path)
        best_model.compile(loss = LOSS, optimizer = OPTIMIZER, metrics=[METRICS[0], tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])
    else:
        raise Exception("Sorry, there are just 3 models")

    # Re-evaluate val_set to check whether this is the best model
    best_model.evaluate(val_set)
    return best_model, model_index


if __name__ == "__main__":
    train_generator, val_generator, test_generator = generate_data(TRAIN_ROOT, VAL_ROOT, TEST_ROOT)
    model_history = train_model(model1, checkpoint1, early1, train_set=train_generator, val_set=val_generator, test_set=test_generator)
    best_model, model_index = load_best_model(1, val_generator)
    best_model.save(SAVED_MODEL_PATH + "\\best_model{}".format(model_index) + ".h5")