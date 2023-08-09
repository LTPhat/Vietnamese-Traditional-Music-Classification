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

![Alt text](https://github.com/LTPhat/Vietnamese-Traditional-Music-Classification/blob/main/model_images/train_phase3.png)

### a) Audio feature extraction
Audio feature extraction is a necessary step in audio signal processing, which is a subfield of signal processing. Different features capture different aspects of sound. Here are some signal domain features.
- ``Time domain:`` These are extracted from waveforms of the raw audio: Zero crossing rate, amplitude envelope, RMS energy ...

- ``Frequency domain:`` Signals are generally converted from the time domain to the frequency domain using the Fourier Transform: Band energy ratio, spectral centroid, spectral flux ...

- ``Time-frequency representation:`` The time-frequency representation is obtained by applying the Short-Time Fourier Transform (STFT) on the time domain waveform: Spectrogram, Mel-spectrogram, constant-Q transform...

In this repo, we extract Mel-spectrogram images from audios of the dataset and feed them to CNN model as an image classification task.

### b) CNN models
We propose 3 models using the extracted mel-spectrogram as input images. With each image, the output vector gives the probability of 5 class.
#### Model 1

![Alt text](https://github.com/LTPhat/Vietnamese-Traditional-Music-Classification/blob/main/model_images/model1/model1_art.png)


#### Model 2

![Alt text](https://github.com/LTPhat/Vietnamese-Traditional-Music-Classification/blob/main/model_images/model2/model2_art.png)


#### Model 3

![Alt text](https://github.com/LTPhat/Vietnamese-Traditional-Music-Classification/blob/main/model_images/model3/model3_art.png)


## Ensemble

In the inference phase, we propose to use late fusion of probabilities, referred to as PROD fusion. Consider predicted probabilities of each model as $\boldsymbol{P_s} = [p_{s1}, p_{s2}, ..., p_{sC}]$  where $C$ is the number of classes and the $s^{th}$  out of  networks evaluated. The predicted probabilities after PROD fusion is obtained by:

$$\boldsymbol{P_{prod}}=[p_1, p_2, ..., p_C], p_i=\frac{1}{S}{\displaystyle \prod_{s=1}^{S}  p_{si}}, 1 \le i \le C$$

Finally, the predicted label $\hat{y}$ is determined by: $\hat{y}=\arg \max{\boldsymbol{P_{prod}}}$

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
- ``Note!`` In order to avoid errors at local when using pydub.AudioSegment, it's better to download ``ffmpeg`` and add them to environment variables. Tutorial here: https://phoenixnap.com/kb/ffmpeg-windows
- Config your own parameters in ``config.py``. Directory configs are available and compatible with the project's folder structure. Hence, it's not recommended to change them.

- Run  ``processing.py``. After running, ``mel-images`` folder contains all the mel-spectrogram images extracted from 5 classes and ``dataset`` folder contains train/val/test folder of images of 5 classes. Constructing the dataset is completed.

- At ``build/train_model.py``, change the model_index to 1, 2, 3 at the last line to train model1, model2 or model3. Then, run this file.
After running, the best model ``.h5`` file will be saved at ``model`` folder. Training is completed.

- Run Streamlit app at ``app/app.py``, upload your new audios and get prediction. The audios uploaded on app will be saved at ``audio_from_user`` folder. Run app using this command:

```sh
streamlit run app/app.py
```

## Some images

![Alt text](https://github.com/LTPhat/Vietnamese-Traditional-Music-Classification/blob/main/app/images/app.png)


![Alt text](https://github.com/LTPhat/Vietnamese-Traditional-Music-Classification/blob/main/app/images/predict.png)


## References

[1] Vietnam Traditional Music (5 genres), https://www.kaggle.com/datasets/homata123/vntm-for-building-model-5-genres.

[2] Librosa Library, https://librosa.org/doc/latest/index.html

[3] TensorFlow, https://www.tensorflow.org/

[4] CHU BA THANH, TRINH VAN LOAN, DAO THI LE THUY, _AUTOMATIC IDENTIFICATION OF SOME VIETNAMESE FOLK
SONGS CHEO AND QUANHO USING DEEP NEURAL NETWORKS_, https://vjs.ac.vn/index.php/jcc/article/view/15961

[5] Valerio Velardo - The Sound of AI, https://www.youtube.com/@ValerioVelardoTheSoundofAI

[6] Dipti Joshi1, Jyoti Pareek, Pushkar Ambatkar, _Comparative Study of Mfcc and Mel Spectrogram for Raga Classification Using CNN_, https://indjst.org/articles/comparative-study-of-mfcc-and-mel-spectrogram-for-raga-classification-using-cnn

[7] Loris Nanni et al, _Ensemble of convolutional neural networks to improve animal audio classification_, https://asmp-eurasipjournals.springeropen.com/articles/10.1186/s13636-020-00175-3
