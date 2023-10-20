# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.11.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# +
### Load necessary libraries ###
import pandas as pd
import numpy as np
import os
import struct
import matplotlib.pyplot as plt
import IPython.display as ipd
import glob
import librosa
import librosa.display
import tensorflow as tf
from tensorflow import keras
from scipy.io import wavfile as wav
from matplotlib.pyplot import specgram
from sklearn.model_selection import KFold

# %matplotlib inline
plt.style.use('ggplot')

data = pd.read_csv("../UrbanSound8K/metadata/UrbanSound8K.csv")

def path_class(filename):
    excerpt = data[data['slice_file_name'] == filename]
    path_name = os.path.join('../UrbanSound8K/audio', 'fold'+str(excerpt.fold.values[0]), filename)
    return path_name, excerpt['class'].values[0]

### Load Sound File ###
def load_sound_file(file_path):
    X,sr = librosa.load(file_path)
    raw_sound = X
    return raw_sound

### Plot Raw Sound Waves and Spectrogram ###
def plot_waves(label,raw_sound):
    fig = plt.figure(figsize=(25,6))
    librosa.display.waveplot(np.array(raw_sound), sr=22050)
    plt.title(label.title())
    plt.show()
    
def plot_specgram(label,raw_sound):
    fig = plt.figure(figsize=(25,6))
    specgram(np.array(raw_sound), Fs=22050)
    plt.title(label.title())
    plt.show()


# -

filepath, label = path_class('7061-6-0-0.wav')
print(path_class('7061-6-0-0.wav'))

raw_sound = load_sound_file(filepath)
print(raw_sound)

plot_waves(label,raw_sound)
plot_specgram(label,raw_sound)

# +
import glob
import librosa
import numpy as np

### Extract Features in Audio File ###
def extract_feature(file_name):
    X, sample_rate = librosa.load(file_name)
    stft = np.abs(librosa.stft(X))
    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T,axis=0)
    chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
    mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T,axis=0)
    contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T,axis=0)
    tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X), sr=sample_rate).T,axis=0)
    return mfccs,chroma,mel,contrast,tonnetz

fccs, chroma, mel, contrast, tonnetz = extract_feature('..\\UrbanSound8K\\audio\\fold1\\7061-6-0-0.wav')
print(fccs, chroma, mel, contrast, tonnetz)


# -

def parse_audio_files(filenames):
    rows = len(filenames)
    features, labels, groups = np.zeros((rows,193)), np.zeros((rows,10)), np.zeros((rows, 1))
    i = 0
    for fn in filenames:
        try:
            mfccs, chroma, mel, contrast,tonnetz = extract_feature(fn)
            ext_features = np.hstack([mfccs,chroma,mel,contrast,tonnetz])
            y_col = int(fn.split('\\')[5].split('-')[1])
            group = int(fn.split('\\')[5].split('-')[0])
        except:
            print(fn)
        else:
            features[i] = ext_features
            labels[i, y_col] = 1
            groups[i] = group
            i += 1
    return features, labels, groups


# +
### Extend files in Every 10 Folds ###
audio_files = []
for i in range(1,11):
    audio_files.extend(glob.glob('..\\UrbanSound8K\\audio\\fold%d\\*.wav' % i))

    
print(len(audio_files))

### Parse each 1000 audio files and save features, labels and groups in urban_sound_0 ~ urban_sound_8 npz files ###
for i in range(9):
    files = audio_files[i*1000: (i+1)*1000]
    X, y, groups = parse_audio_files(files)
    for r in y:
        if np.sum(r) > 1.5:
            print('error occured')
            break
    np.savez('../UrbanSound8K/npfiles/urban_sound_%d' % i, X=X, y=y, groups=groups)

# +
import glob
import numpy as np

### Append every feature, label and group in urban_sound file ###
### [[File1_Features] [File2_Features] ... [File8732_Features]
###  [File1_Label]    [File2_Label]   ...  [File8732_Label]
###  [File1_Group]    [File2_Group]  ...   [File8732_Group]]
X = np.empty((0, 193))
y = np.empty((0, 10))
groups = np.empty((0, 1))
npz_files = glob.glob('../UrbanSound8K/npfiles/urban_sound_?.npz')
for fn in npz_files:
    print(fn)
    data = np.load(fn)
    X = np.append(X, data['X'], axis=0)
    y = np.append(y, data['y'], axis=0)
    groups = np.append(groups, data['groups'], axis=0)

print(groups[groups>0])

### Shape of Features and Labels ###
print(X.shape, y.shape)
for r in y:
    if np.sum(r) > 1.5:
        print(r)
np.savez('../UrbanSound8K/npfiles/urban_sound', X=X, y=y, groups=groups)

print(X[0])
print(y[0])
print(groups[0])

# +
### Test Train Set Split ###
import numpy as np
from sklearn.model_selection import GroupShuffleSplit

sound_data = np.load('../UrbanSound8K/npfiles/urban_sound.npz')
X_data = sound_data['X']
y_data = sound_data['y']
groups = sound_data['groups']

print(groups[groups > 0])

gss = GroupShuffleSplit(n_splits=1, test_size=0.2)
for train_idx, test_idx in gss.split(X_data, y_data, groups=groups):
    X_train = X_data[train_idx]
    y_train = y_data[train_idx]
    groups_train = groups[train_idx]

    X_test = X_data[test_idx]
    y_test = y_data[test_idx]
    groups_test = groups[test_idx]
    
    print(X_train.shape, X_test.shape)
    
np.savez('../UrbanSound8K/npfiles/urban_sound_train', X=X_train, y=y_train, groups=groups_train)
np.savez('../UrbanSound8K/npfiles/urban_sound_test', X=X_test, y=y_test, groups=groups_test)

# +
### Train Data ###
import tensorflow as tf
from tensorflow import keras

train_data = np.load('../UrbanSound8K/npfiles/urban_sound_train.npz')
X_train = train_data['X']
y_train = train_data['y']
groups_train = train_data['groups']
X_train.shape, y_train.shape, groups_train.shape


# -

### Define feedforward network architecture ###
def get_network():
    input_shape = (193,)
    num_classes = 10
    keras.backend.clear_session()
    
    model = keras.models.Sequential()
    model.add(keras.layers.Dense(256, activation="relu", input_shape=input_shape))
    model.add(keras.layers.Dense(128, activation="relu", input_shape=input_shape))
    model.add(keras.layers.Dense(64, activation="relu", input_shape=input_shape))
    model.add(keras.layers.Dense(num_classes, activation = "softmax"))
    model.compile(optimizer=keras.optimizers.Adam(1e-4), 
        loss=tf.keras.losses.CategoricalCrossentropy(), 
        metrics=["accuracy"])
    
    return model


# +
### Learn ###
model = get_network()
model.summary()
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs = 50, batch_size = 24, verbose = 0)

l, a = model.evaluate(X_test, y_test, verbose = 0)
print("Loss: {0} | Accuracy: {1}".format(l, a))
# -

### Predict ###
print(y_test[40])
X_predict = X_test[40:41]
y_predict = model.predict(X_predict, verbose=0)
print(y_predict)
np.argmax(y_predict[0])

# +
y_vloss = history.history['val_loss']
y_loss = history.history['loss']

x_len = np.arange(len(y_loss))
plt.plot(x_len, y_vloss, marker='.', c='red', label="Validation-set Loss")
plt.plot(x_len, y_loss, marker='.', c='blue', label="Train-set Loss")

plt.legend(loc='upper right')
plt.grid()
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()

# +
model = get_network()
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs = 10, batch_size = 24, verbose = 0)

l, a = model.evaluate(X_test, y_test, verbose = 0)
print("Loss: {0} | Accuracy: {1}".format(l, a))
# -

### Predict ###
print(y_test[40])
X_predict = X_test[40:41]
y_predict = model.predict(X_predict, verbose=0)
print(y_predict)
print(np.mean(y_predict, axis=0))
np.argmax(y_predict[0])


