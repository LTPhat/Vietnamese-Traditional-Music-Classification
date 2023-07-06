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



class Preprocessing():
    def __init__(self, root, samples, type_index):
        self.root = root            # Directory of folder
        self.samples = samples      # Dictionary store information of each class folder
        self.type_index = type_index


    def _load_dir_samples(self, num_of_samples, mode):
        """
        Load directory into 'samples' dictionary
        - Random: Load random dir
        - All: Load all dir

        Return:
        - Samples: Dictionary {index: {"dir": "/...."}}
        """
        def padding(index):
            # Padding
            if 0 <= index < 10:
                index = "00" +str(index)
            elif 10 <= index < 100:
                index = "0" +str(index)
            return index
        for i in range(0, num_of_samples):
            if mode == "random": # Mode load random samples
                random_index = np.random.randint(0, 500)
                index = random_index
                self.samples[index] = {}  # For futher append values
                random_index = padding(random_index)
                self.samples[index]["dir"] = self.root + "\\"  + type_list[self.type_index][0] + "\\" + type_list[self.type_index][1] + "." + str(random_index) + ".wav"
                # self.samples_list[index]["dir"] = (os.path.join(self.root, type_list[self.type_index][0], type_list[self.type_index][1] + "." + str(random_index) + ".wav"))
            
            if mode == "all":  # Mode load all samples
                index = i
                self.samples[index] = {}
                i = padding(i)
                self.samples[index]["dir"] = self.root + "\\"  + type_list[self.type_index][0] + "\\" + type_list[self.type_index][1] + "." + str(i) + ".wav"
                # self.samples[index]["dir"] = os.path.join(self.root, type_list[self.type_index][0], type_list[self.type_index][1] + "." + str(i) + ".wav")

        return self.samples


    def _load_samples(self):
        """
        Load and sampling
        Input: samples_listdir - Dictionary {index: {"dir": "/...."}}
        Output: samples_listdir - Dictionary {index: {"dir": "/....", "sampling": array}}
        """
        for index, sample in self.samples.items():
            file, sr = lb.load(sample["dir"])
            if len(self.samples[index]) == 1:  # Avoid adding multiple times
                self.samples[index]["sampling"] = file
        return self.samples
    

    def _get_fft(self, n_fft, hop_length):
        """
        Input: samples: {index: {"dir": "/..."}}
        Output: samples: {index: {"dir": "/...", "stft:" array}}
        """
        for index, item in self.samples.items():
            # Get STFT
            D = np.abs(lb.stft(item["sampling"], n_fft = n_fft, hop_length = hop_length))
            self.samples[index]["stft"] = D
        return self.samples


    def _get_mel_spectrogram(self, sr):
        """
        Get log-mel-spectrogram (db)
        Input: {index: {"dir": "/...", "sampling": array, "stft": array, }}
        Output: {index: {"dir": "/...", "sampling": array, "stft": array, "mel-spec-db": array}}
        """
        for index, item in self.samples.items():
            S = lb.feature.melspectrogram(y = item["sampling"], sr = sr)
            S_db = lb.amplitude_to_db(S, ref=np.max)
            self.samples[index]["mel-spec-db"] = S_db
        return self.samples

    def _save_mel_spec(self, save_root):
        """
        Save log-mel-spec
        After running, images of a class will be saved in : root/class/file_name.png
        """

        for _, item in self.samples.items():
            S_db = item["mel-spec-db"]
            folder_root = save_root + "\\" + type_list[self.type_index][0]
            print(folder_root)
            if not os.path.exists(folder_root):
                os.makedirs(folder_root)
                print("Makedir")
            # Get file name from fir
            file_name = item["dir"].split("\\")[-1][:-4]
            plt.imsave(folder_root + "\\{}".format(file_name) + ".png", S_db)





five_samples = Preprocessing(RAW_ROOT, {}, 0)
print(five_samples._load_dir_samples(num_of_samples = 5, mode ="random"))
print(five_samples._load_samples())
print(five_samples._get_fft(N_FFT, HOP_LENGTH))
print(five_samples._get_mel_spectrogram(SR))
five_samples._save_mel_spec("test_save_images") # Change to "mel-images" FOLDER_ROOT
print("done")


