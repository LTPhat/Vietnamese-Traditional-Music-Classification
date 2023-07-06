# Define directories 

RAW_ROOT = '/VN-music-classfication/rawdata'
FOLDER_ROOT = "/VN-music-classfication/mel-images"
DATASET_ROOT = "/VN-music-classfication/dataset"
TRAIN_ROOT = "/VN-music-classfication/dataset/train"
VAL_ROOT = "/VN-music-classfication/dataset/val"
TEST_ROOT = "/content/drive/MyDrive/DATA/test"
CHECKPOINT_FILEPATH = '/VN-music-classfication/checkpoint'
SAVED_MODEL_PATH = '/VN-music-classfication/model'
TEST_AUDIO_PATH = "/VN-music-classfication/test_audio"

# Define global variable

type_list = {0: ["cailuong", "CaiLuong"], 1: ["catru", "Catru"], 2:["chauvan", "Chauvan"], 3: ["cheo", "Cheo"], 4: ["hatxam", "Xam"]}

class_list = {0: "cailuong", 1: "catru", 2:"chauvan", 3: "cheo", 4: "hatxam"}

# Define processing parameters
SR = 22050
N_FFT = 2048
HOP_LENGTH = 512

# Train/Val/Test split

TRAIN_RATE = 0.75
VAL_RATE = 0.15
TEST_RATE = 0.1

# Input/ Output
N_CLASS = 5
INPUT_SHAPE = (128, 1292)

