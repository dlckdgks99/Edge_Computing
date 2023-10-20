# Urban-Sound-Classification
UrbanSound8K Classification with Deep Learning Model (FFNN, CNN, RNN)

## Environment
- Create a root folder to work and clone [Urban Sound Classification](https://github.com/lucy0828/Urban-Sound-Classification) repository inside the root folder
- Download UrbanSound8K Dataset in the root folder
- Anaconda Environment

  ```
  conda create -n audio python=3.7
  activate audio
  ```
  Python 3.5 (or above) is used during development and following libraries are required to run the code provided in the notebook:
    * Tensorflow 2.x
    * Numpy
    * Matplotlib
    * Librosa
    * Kapre

## UrbanSound8K Dataset
The UrbanSound8k dataset used for model training, can be downloaded from the following [link](https://serv.cusp.nyu.edu/projects/urbansounddataset/urbansound8k.html).
- Create `"clean"` `"npfiles"` `"wavfiles"` `"test"` `"train"` folders inside UrbanSound8K folder 
- Create folders from `"0"` to `"9"` in each `"clean"` `"wavfiles"` `"test"` `"train"` folders
- You can find wav files in `"audio"` folder
- You can find UrbanSound8K csv file in `"metadata"` folder
- UrbanSound8K example "gun shot" signal and spectrogram plots are below
<p align="center">
  <img src="https://user-images.githubusercontent.com/46836844/113685060-d38d9e80-9700-11eb-8ba1-1797d77fde13.png">
</p>

## Jupyter Notebooks
Assuming you have ipykernel installed from your conda environment
  ```
  ipython kernel install --user --name=audio
  conda activate audio
  jupyter-notebook
  ```

## UrbanSound8K Analysis
[UrbanSound8K Analysis.py](https://github.com/lucy0828/Urban-Sound-Classification/blob/master/UrbanSound8K%20Analysis.py) is a Notebook script to analyze UrbanSound8K :
- Meta-data
- Class distribution
- Wav file plot
- Sampling rate, bit depth, channels

## Feedforward Neural Network
[FFNN.py](https://github.com/lucy0828/Urban-Sound-Classification/blob/master/FFNN.py) is based on this [blogpost](http://aqibsaeed.github.io/2016-09-03-urban-sound-classification-part-1/) 
to classify urban sounds using Feedforward Neural Network
### Feature Extraction
Various features of sound are extracted with Librosa library and saved as numpy array with corresponding labels in `"npfiles"` folder
- melspectrogram: Compute a mel-scaled power spectrogram
- mfcc: Mel-frequency cepstral coefficients
- chorma-stft: Compute a chromagram from a waveform or power spectrogram
- spectral_contrast: Compute spectral contrast
- tonnetz: Computes the tonal centroid features (tonnetz)
### Classifier
Test and train sets are splitted and saved in `"npfiles"` foler. Keras feedforward neural network are used to train the model.
<p align="center">
  <img src="https://user-images.githubusercontent.com/46836844/113801125-2748c800-9793-11eb-9e98-089ff4810a14.png">
</p>

## CNN and RNN
This project is based on this [repository](https://github.com/seth814/Audio-Classification) to classify urban sounds using Convolution and Recurrent Neural Networks
### Audio Preprocessing 
[Data Process.py](https://github.com/lucy0828/Urban-Sound-Classification/blob/master/Data%20Process.py) preprocess audio data with `test_threshold(args)` and `split_wavs(args)` functions
- Files in `"audio"` folder are grouped in same classes and saved in `"wavfiles"` folder
- Low magnitude wavfiles data below threshold are removed by signal envelop function
- Wavfiles are splitted in delta-time(1 sec) samples and saved in `"clean"` folder
<p align="center">
  <img src="https://user-images.githubusercontent.com/46836844/113805041-b4434f80-979a-11eb-9445-5d98ea3138c1.png">
</p>

### Feature Extraction
[Models.py](https://github.com/lucy0828/Urban-Sound-Classification/blob/master/Models.py) uses [Kapre](https://kapre.readthedocs.io/en/latest/) to construct input layer of Keras model
- [get_melspectrogram_layer](https://kapre.readthedocs.io/en/latest/composed.html): Returns a melspectrogram input layer which consists of STFT, Magnitude, Filterbank, and MagtoDec
<p align="center">
  <img src="https://user-images.githubusercontent.com/46836844/113815059-8ddadf80-97ad-11eb-948b-44c24b147046.png">
</p>

### Classifier
- Conv1D, Conv2D, LSTM Model Summary
<p align="center">
  <img src="https://user-images.githubusercontent.com/46836844/113815947-f5455f00-97ae-11eb-88d8-0feba535a9a0.png">
</p>

### Train
[Train.py](https://github.com/lucy0828/Urban-Sound-Classification/blob/master/Train.py) trains selected model(conv1d, conv2d, lstm) by 30 epochs
- Trained models are saved in `"models"` folder
- Train and test set accuracy history are saved in `"logs"` folder
<p align="center">
  <img src="https://user-images.githubusercontent.com/46836844/113816892-6fc2ae80-97b0-11eb-8adf-2ae2017ee202.png">
</p>

### Predict
[Predict.py](https://github.com/lucy0828/Urban-Sound-Classification/blob/master/Predict.py) calculates total mean of samples' prediction scores in wavfile and predicts class
- `"make_prediction(args)"`: Predicts one wavfile class from selected model
- `"make_predictions(args)"`: Predicts all wavfiles classes from selected model
<p align="center">
  <img src="https://user-images.githubusercontent.com/46836844/113818057-34c17a80-97b2-11eb-99bf-735587543b73.png">
</p>
