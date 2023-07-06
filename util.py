import os
import librosa as lb
import numpy as np
# import IPython.display as ipd
import matplotlib.pyplot as plt
import random
import numpy as np
from config import *
from sklearn.model_selection import train_test_split
import shutil


def plot_waveform(samples, type_index, sr = SR):
    """
    Waveform plot of samples
    """
    color = random.choice(["blue", "red", "yellow", "brown", "purple"])
    for index, sample in samples.items():
      plt.figure(figsize = (16, 5))
      lb.display.waveshow(y = sample["sampling"], sr = sr, color = color);
      plt.title("Sound Waves of sample {} of class {}".format(index, type_list[type_index][0]), fontsize = 23);


def plot_fft(samples, type_index):
    """
    Get frequency domain representation
    """
    for index, item in samples.items():
        plt.figure(figsize = (16, 6))
        plt.plot(item["stft"])
        plt.xlabel("Time")
        plt.ylabel("Frequency")
        plt.title("STFT of sample {} of class {}".format(index, type_list[type_index][0]))


def plot_spectrogram(samples, type_index, hop_length):
    """
    Plot spectrogram
    """
    for index, item in samples.items():
        DB = lb.amplitude_to_db(item["stft"], ref = np.max)
        plt.figure(figsize = (25, 10))
        lb.display.specshow(DB, hop_length= hop_length, x_axis = "time", y_axis = "log")
        plt.title("Spectrogram of sample {} of class {}".format(index, type_list[type_index][0]), fontsize = 20)
        plt.colorbar()


def train_val_test_split(folder_root, dataset_root, type_index):
    """
    Split and save train/val/test set
    Input:
    - folder_root: folder_root containing mel-spec images
    - dataset_root: Directory to save dataset
    - type_root : train_root, val_root or test_root
    - type_index: class index in type_list
    """

    def save_set(subset, dataset_root, typeset, type_index):
      """
      Save X_train, X_val, X_test to their respective dir
      Input:
        - subset - X_train, X_val, X_test
        - dataset_root: Directory to save dataset
        - typeset - train, val, test
        - type index - Class index
      """
      # Copy file from subset to train/val/test folder
      for file in subset:
          srcpath = os.path.join(src_dir, file)
          dst_dir = dataset_root + "/" + typeset + "/{}".format(type_list[type_index][0])
          if not os.path.exists(dst_dir):
              os.makedirs(dst_dir)
          shutil.copy(srcpath, dst_dir)


    src_dir = folder_root + "/{}".format(type_list[type_index][0])
    X = os.listdir(src_dir)
    Y = ["{}".format(type_list[type_index][0]) for i in range(0, len(X))]
    # Train 75%, test 25%
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size= 1 - TRAIN_RATE, random_state=42, shuffle = True)
    # Val 15 %, test 10%
    X_val, X_test, Y_val, Y_test = train_test_split(X_test, Y_test, test_size = TEST_RATE / (TEST_RATE + VAL_RATE), random_state=42, shuffle = True)

    # Create dataset_root to save dataset
    if not os.path.exists(dataset_root):
        os.makedirs(dataset_root)
    # Save train/val/test of each class
    save_set(X_train, dataset_root, "train", type_index)
    save_set(X_val, dataset_root, "val", type_index)
    save_set(X_test, dataset_root, "test", type_index)
