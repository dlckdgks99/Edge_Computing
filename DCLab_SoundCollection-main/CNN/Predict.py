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
from tensorflow.keras.models import load_model
from kapre.time_frequency import STFT, Magnitude, ApplyFilterbank, MagnitudeToDecibel
from sklearn.preprocessing import LabelEncoder
import numpy as np
from glob import glob
import argparse
import os
import pandas as pd
from tqdm import tqdm
import librosa

def envelope(y, rate, threshold):
    mask = []
    y = pd.Series(y).apply(np.abs)
    y_mean = y.rolling(window=int(rate/20),
                       min_periods=1,
                       center=True).max()
    for mean in y_mean:
        if mean > threshold:
            mask.append(True)
        else:
            mask.append(False)
    return mask, y_mean

'''
Make predictions for multiple files
'''    
def make_predictions(args):

    print('Make Predictions')
    
    model = load_model(args.model_fn,
        custom_objects={'STFT':STFT,
                        'Magnitude':Magnitude,
                        'ApplyFilterbank':ApplyFilterbank,
                        'MagnitudeToDecibel':MagnitudeToDecibel})
    wav_paths = glob('{}/**'.format(args.src_dir), recursive=True)
    wav_paths = sorted([x.replace(os.sep, '/') for x in wav_paths if '.wav' in x])
    classes = sorted(os.listdir(args.src_dir))
    labels = [os.path.split(x)[0].split('/')[-1] for x in wav_paths]
    le = LabelEncoder()
    y_true = le.fit_transform(labels)
    results = []

    for z, wav_fn in enumerate(wav_paths):
        # print(wav_fn)
        wav, rate = librosa.load(wav_fn, args.sr)
        mask, env = envelope(wav, rate, threshold=args.threshold)
        clean_wav = wav[mask]
        step = int(args.sr*args.dt)
        batch = []

        for i in range(0, clean_wav.shape[0], step):
            sample = clean_wav[i:i+step]
            sample = sample.reshape(-1, 1)
            if sample.shape[0] < step:
                tmp = np.zeros(shape=(step, 1), dtype=np.float32)
                tmp[:sample.shape[0],:] = sample.flatten().reshape(-1, 1)
                sample = tmp
            batch.append(sample)
        X_batch = np.array(batch, dtype=np.float32)
        y_pred = model.predict(X_batch)
        y_mean = np.mean(y_pred, axis=0)
        y_pred = np.argmax(y_mean)
        real_class = os.path.dirname(wav_fn).split('/')[-1]
        if y_pred == 0:
            os.remove(wav_fn)
        # print('\n')
        results.append(y_mean)

    # np.save(os.path.join('./logs', args.pred_fn), np.array(results))


# -

# if __name__ == '__main__':

#     parser = argparse.ArgumentParser(description='Audio Classification Training')
#     parser.add_argument('--model_fn', type=str, default='./models/conv1d.h5',
#                         help='model file to make predictions')
#     parser.add_argument('--pred_fn', type=str, default='y_pred',
#                         help='fn to write predictions in logs dir')
#     parser.add_argument('--src_dir', type=str, default='./UrbanSound8K/test',
#                         help='directory containing wavfiles to predict')
#     parser.add_argument('--fn', type=str, default='210428-1-0-2.wav',
#                         help='file name to predict')
#     parser.add_argument('--dt', type=float, default=1.0,
#                         help='time in seconds to sample audio')
#     parser.add_argument('--sr', type=int, default=16000,
#                         help='sample rate of clean audio')
#     parser.add_argument('--threshold', type=str, default=0.003,
#                         help='threshold magnitude for np.int16 dtype')
#     args, _ = parser.parse_known_args()

#     make_predictions(args)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Audio Classification Training')
    parser.add_argument('--model_fn', type=str, default='./models/conv2d.h5',
                        help='model file to make predictions')
    parser.add_argument('--pred_fn', type=str, default='y_pred',
                        help='fn to write predictions in logs dir')
    parser.add_argument('--src_dir', type=str, default='./UrbanSound8K/test',
                        help='directory containing wavfiles to predict')
    parser.add_argument('--fn', type=str, default='210428-1-0-2.wav',
                        help='file name to predict')
    parser.add_argument('--dt', type=float, default=1.0,
                        help='time in seconds to sample audio')
    parser.add_argument('--sr', type=int, default=16000,
                        help='sample rate of clean audio')
    parser.add_argument('--threshold', type=str, default=0.003,
                        help='threshold magnitude for np.int16 dtype')
    args, _ = parser.parse_known_args()

    make_predictions(args)

# if __name__ == '__main__':

#     parser = argparse.ArgumentParser(description='Audio Classification Training')
#     parser.add_argument('--model_fn', type=str, default='./models/lstm.h5',
#                         help='model file to make predictions')
#     parser.add_argument('--pred_fn', type=str, default='y_pred',
#                         help='fn to write predictions in logs dir')
#     parser.add_argument('--src_dir', type=str, default='../UrbanSound8K/test',
#                         help='directory containing wavfiles to predict')
#     parser.add_argument('--fn', type=str, default='210428-1-0-2.wav',
#                         help='file name to predict')
#     parser.add_argument('--dt', type=float, default=1.0,
#                         help='time in seconds to sample audio')
#     parser.add_argument('--sr', type=int, default=16000,
#                         help='sample rate of clean audio')
#     parser.add_argument('--threshold', type=str, default=0.003,
#                         help='threshold magnitude for np.int16 dtype')
#     args, _ = parser.parse_known_args()

#     make_predictions(args)

# if __name__ == '__main__':

#     parser = argparse.ArgumentParser(description='Audio Classification Training')
#     parser.add_argument('--model_fn', type=str, default='./models/conv1d.h5',
#                         help='model file to make predictions')
#     parser.add_argument('--pred_fn', type=str, default='y_pred',
#                         help='fn to write predictions in logs dir')
#     parser.add_argument('--src_dir', type=str, default='../UrbanSound8K/recorded',
#                         help='directory containing wavfiles to predict')
#     parser.add_argument('--fn', type=str, default='210428-1-0-2.wav',
#                         help='file name to predict')
#     parser.add_argument('--dt', type=float, default=1.0,
#                         help='time in seconds to sample audio')
#     parser.add_argument('--sr', type=int, default=16000,
#                         help='sample rate of clean audio')
#     parser.add_argument('--threshold', type=str, default=0.003,
#                         help='threshold magnitude for np.int16 dtype')
#     args, _ = parser.parse_known_args()

#     make_predictions(args)


