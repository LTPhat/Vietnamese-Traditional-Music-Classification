# Vietnamese Traditional Music Classification
- **Digital Signal Processing Course Project (EE2015).**
- **Audio Classification using Mel Spectrograms and Convolution Neural Networks.**
- **Finished:** ```09/07/2023```

## Dataset
- The dataset includes audio files of 5 classes:  ``cailuong``,  ``catru``,  ``chauvan``,  ``cheo``,  ``hatxam``.
- Each class includes 500 wav files with a length of about 30s.
- ``Vietnam Traditional Music (5 genres):`` https://www.kaggle.com/datasets/homata123/vntm-for-building-model-5-genres.
- Download the dataset, create a folder named ``rawdata`` in the project's folder and configure the dataset as shown below.
  
      ...
      ├── model_images
      ├── notebook
      ├── rawdata                   
      │  ├── cailuong 
      |  |      ├── Cailuong000.wav
      |  |      ├── Cailuong001.wav
      |  |      ├── Cailuong002.wav   
      |  |      ├── ...
      │  ├── catru    
      |  |      ├── Catru000.wav
      |  |      ├── Catru001.wav
      |  |      ├── Catru002.wav   
      |  |      ├── ...        
      │  ├── chauvan  
      |  |      ├── Chauvan000.wav
      |  |      ├── Chauvan001.wav
      |  |      ├── Chauvan002.wav   
      |  |      ├── ...   
      │  ├── cheo  
      |  |      ├── Cheo000.wav
      |  |      ├── Cheo001.wav
      |  |      ├── Cheo002.wav   
      |  |      ├── ...  
      │  ├── hatxam  
      |  |      ├── Hatxam000.wav
      |  |      ├── Hatxam001.wav
      |  |      ├── Hatxam002.wav   
      |  |      ├── ...              
      ├── test_audio
      ...

## Workflow
The project's workflow is illustrated in the figure below:

![Alt text](https://github.com/LTPhat/Vietnamese-Traditional-Music-Classification/blob/main/model_images/train_phase2.png)

### a) Audio feature extraction
Audio feature extraction is a necessary step in audio signal processing, which is a subfield of signal processing. Different features capture different aspects of sound. Here are some signal domain features.
- ``Time domain:`` These are extracted from waveforms of the raw audio: Zero crossing rate, amplitude envelope, RMS energy ...
- ``Frequency domain:`` Signals are generally converted from the time domain to the frequency domain using the Fourier Transform: Band energy ratio, spectral centroid, spectral flux ...
- ``Time-frequency representation:`` The time-frequency representation is obtained by applying the Short-Time Fourier Transform (STFT) on the time domain waveform: Spectrogram, Mel-spectrogram, constant-Q transform...

In this repo, we extract Mel-spectrogram images from audios of the dataset and feed them to CNN model as an image classification task.

### b) CNN models
We propose 3 models using the extracted mel-spectrogram as input images. With each image, the output vector gives the probability of 5 class.
#### Model 1

![Alt text](https://github.com/LTPhat/Vietnamese-Traditional-Music-Classification/blob/main/model_images/model1/model1.png)

> Evaluation:

```sh
val_set: loss: 0.4680 - accuracy: 0.8960 - precision: 0.8954 - recall: 0.8907
test_set: loss: 0.4514 - accuracy: 0.8840 - precision: 0.8984 - recall: 0.8840
```

#### Model 2

![Alt text](https://github.com/LTPhat/Vietnamese-Traditional-Music-Classification/blob/main/model_images/model2/model2.png)

> Evaluation:

```sh
val_set: loss: 0.6250 - accuracy: 0.8720 - precision_1: 0.8767 - recall_1: 0.8720
test_set: loss: 0.4261 - accuracy: 0.9000 - precision_1: 0.9032 - recall_1: 0.8960
```

#### Model 3

![Alt text](https://github.com/LTPhat/Vietnamese-Traditional-Music-Classification/blob/main/model_images/model3/model3.png)

```sh
val_set: loss: 0.3558 - accuracy: 0.9093 - precision_1: 0.9212 - recall_1: 0.9040
test_set: loss: 0.2625 - accuracy: 0.9320 - precision_1: 0.9357 - recall_1: 0.9320
```

## Tutorial

To run the code of this project, please follow these steps:
- Install required libraries, dependencies.
```sh
numpy
librosa
tensorflow
matplotlib
pydub
sklearn
seaborn
```

- Config your own parameters in ``config.py``. Directory configs are available and compatible with the project's folder structure. Hence, it's not recommended to change them.

- Run  ``processing.py``. After running, ``mel-images`` folder contains all the mel-spectrogram images extracted from 5 class and ``dataset`` folder contains train/val/test folder of images of 5 class. Constructing the dataset is completed.

- At ``build/train_model.py``, change the model_index to 1, 2, 3 at the last line to train model1, model2 or model3. Then, run this file.
After running, the best model ``.h5`` file will be saved at ``model\``
