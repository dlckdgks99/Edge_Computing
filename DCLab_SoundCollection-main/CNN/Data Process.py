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
'''
group files in different folders in same class directory
'''
import os
import fnmatch
import shutil

for dirpath, dirnames, files in os.walk('./UrbanSound8K/audio'):
    for filename in files:
        for i in range(0,10):
            if fnmatch.fnmatch(filename, '*-'+str(i)+'-*-*.wav'):
                shutil.copy2(dirpath+'/'+filename, './UrbanSound8K/wavfiles/'+str(i)+'/')

# +
'''
count sample files in each class folders
split train and test data
'''
count = 0
for i in range(0,10):
    list = os.listdir("./UrbanSound8K/wavfiles/" + str(i)) # dir is your directory path
    print("./UrbanSound8K/wavfiles/" + str(i) + ": " + str(len(list)))
    count = count + len(list)
    
    trainlist = 8*int(len(list)/10)

    for j in range(0,trainlist):
        shutil.copy("./UrbanSound8K/wavfiles/" + str(i) + "/" + list[j], "./UrbanSound8K/train/" + str(i))
    for j in range(trainlist,len(list)):
        shutil.copy("./UrbanSound8K/wavfiles/" + str(i) + "/" + list[j], "./UrbanSound8K/test/" + str(i))

print("Total sample files: " + str(count))

count = 0
for i in range(0,10):
    list = os.listdir("./UrbanSound8K/train/" + str(i)) # dir is your directory path
    print("./UrbanSound8K/train/" + str(i) + ": " + str(len(list)))
    count = count + len(list)

print("Total sample files: " + str(count))

count = 0
for i in range(0,10):
    list = os.listdir("./UrbanSound8K/test/" + str(i)) # dir is your directory path
    print("./UrbanSound8K/test/" + str(i) + ": " + str(len(list)))
    count = count + len(list)

print("Total sample files: " + str(count))
# +
'''
Functions used for processing audio files
'''
import matplotlib.pyplot as plt
from scipy.io import wavfile
import argparse
from glob import glob
import numpy as np
import pandas as pd
from tqdm import tqdm
import wavio
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


def save_sample(sample, rate, target_dir, fn, ix):
    fn = fn.split('.wav')[0]
    dst_path = os.path.join(target_dir, fn+'_{}.wav'.format(str(ix)))
    print(dst_path)
    if os.path.exists(dst_path):
        return
    wavfile.write(dst_path, rate, sample)


def check_dir(path):
    if os.path.exists(path) is False:
        os.mkdir(path)


def split_wavs(args):
    src_root = args.src_root
    dst_root = args.dst_root
    dt = args.delta_time

    wav_paths = glob('{}/**'.format(src_root), recursive=True)
    wav_paths = [x for x in wav_paths if '.wav' in x]
    dirs = os.listdir(src_root)
    check_dir(dst_root)
    classes = os.listdir(src_root)
    for _cls in classes:
        target_dir = os.path.join(dst_root, _cls)
        check_dir(target_dir)
        src_dir = os.path.join(src_root, _cls)
        for fn in tqdm(os.listdir(src_dir)):
            src_fn = os.path.join(src_dir, fn)
            wav, rate = librosa.load(src_fn, args.sr)
            mask, y_mean = envelope(wav, rate, threshold=args.threshold)
            wav = wav[mask]
            delta_sample = int(dt*rate)

            '''
            cleaned audio is less than a single sample
            pad with zeros to delta_sample size
            '''
            if wav.shape[0] < delta_sample:
                sample = np.zeros(shape=(delta_sample,), dtype=np.int16)
                sample[:wav.shape[0]] = wav
                save_sample(sample, rate, target_dir, fn, 0)
            # step through audio and save every delta_sample
            # discard the ending audio if it is too short
            else:
                trunc = wav.shape[0] % delta_sample
                for cnt, i in enumerate(np.arange(0, wav.shape[0]-trunc, delta_sample)):
                    start = int(i)
                    stop = int(i + delta_sample)
                    sample = wav[start:stop]
                    save_sample(sample, rate, target_dir, fn, cnt)


def test_threshold(args):
    src_root = args.src_root
    wav_paths = glob('{}/**'.format(src_root), recursive=True)
    wav_path = [x for x in wav_paths if args.fn in x]
    print(wav_paths)
    if len(wav_path) != 1:
        print('audio file not found for sub-string: {}'.format(args.fn))
        return
    wav, rate = librosa.load(wav_path[0], args.sr)
    mask, env = envelope(wav, rate, threshold=args.threshold)
    plt.style.use('ggplot')
    plt.title('Signal Envelope, Threshold = {}'.format(str(args.threshold)))
    plt.plot(wav[mask], color='c', label='keep')
    plt.plot(env, color='m', label='envelope')
    plt.plot(wav[np.logical_not(mask)], color='r', label='remove')
    plt.grid(False)
    plt.legend(loc='best')
    plt.show()


# -

'''
set main arguments 
test and set threshold
split audio files in samples with delta time 
'''
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Cleaning audio data')
    parser.add_argument('--src_root', type=str, default='./UrbanSound8K/test',
                        help='directory of audio files in total duration')
    parser.add_argument('--dst_root', type=str, default='./UrbanSound8K/clean',
                        help='directory to put audio files split by delta_time')
    parser.add_argument('--delta_time', '-dt', type=float, default=1.0,
                        help='time in seconds to sample audio')
    parser.add_argument('--sr', type=int, default=16000,
                        help='rate to downsample audio')

    parser.add_argument('--fn', type=str, default='7383-3-0-1.wav',
                        help='file to plot over time to check magnitude')
    parser.add_argument('--threshold', type=str, default=0.003,
                        help='threshold magnitude for np.float32 dtype')
    args, _ = parser.parse_known_args()

    #test_threshold(args)

    split_wavs(args)




