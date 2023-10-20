import librosa
import soundfile as sf
import os
import time
from pydub import AudioSegment
import scipy.io.wavfile as wavfile

from CNN.Predict import make_predictions as recogAudio
from AECNN.AECNN.test_aecnn import cleanAudio
import argparse

def splitAudio(argsMain):
    audio_file = argsMain.data_dir + argsMain.data_name
    sr, y = wavfile.read(audio_file)
    print("Start Split Audio")
    for i in range(3600):
        start = i
        end = i + 1
        ny = y[sr*start:sr*end]
        sf.write(argsMain.data_dir+"split/"+str(i)+".wav", ny, sr)
    print("End Split Audio")

def sumAudio(path):
    splitPath = path + 'split/'
    sumPath = path + 'sum/'
    file_list = os.listdir(splitPath) #경로 읽어 파일명 리스트 만들기
    li = []
    for i in file_list:
        li.append(i.replace(".wav",""))
    
    li.sort(key=int)
    beforeNum = int(li[0])
    nowSound = AudioSegment.from_wav(splitPath + li[0] + ".wav")
    n = 0
    for i in li:
        num = int(i)
        if num == beforeNum + 1:
            nowSound += AudioSegment.from_wav(splitPath + i  + ".wav")
            beforeNum = num
        elif num == beforeNum:
            pass
        else:
            nowSound.export(sumPath + str(n) + ".wav", format="wav")
            nowSound = AudioSegment.from_wav(splitPath + i + ".wav")
            beforeNum = num
            n += 1 
    
    print("number of output files :",n)
            
    
if __name__ == '__main__':
    start = time.time()  # 시작 시간 저장
    # arguments for split
    parserMain = argparse.ArgumentParser(description='Pre Setting')
    parserMain.add_argument('--data_dir', type=str, default='./data/',
                        help='data_dir')
    parserMain.add_argument('--data_name', type=str, default='exp/expData1.wav',
                        help='data_dir')
    argsMain, _ = parserMain.parse_known_args()
    
    #argument for Sound Recognition
    parserCNN = argparse.ArgumentParser(description='Audio Classification Training')
    parserCNN.add_argument('--model_fn', type=str, default='./CNN/models/conv2d.h5',
                        help='model file to make predictions')
    parserCNN.add_argument('--pred_fn', type=str, default='y_pred',
                        help='fn to write predictions in logs dir')
    parserCNN.add_argument('--src_dir', type=str, default='./data/split',
                        help='directory containing wavfiles to predict')
    parserCNN.add_argument('--fn', type=str, default='210428-1-0-2.wav',
                        help='file name to predict')
    parserCNN.add_argument('--dt', type=float, default=1.0,
                        help='time in seconds to sample audio')
    parserCNN.add_argument('--sr', type=int, default=16000,
                        help='sample rate of clean audio')
    parserCNN.add_argument('--threshold', type=str, default=0.003,
                        help='threshold magnitude for np.int16 dtype')
    argsCNN, _ = parserCNN.parse_known_args()

    print("####################")
    print("Process PID",os.getpid())
    print("####################")
    time.sleep(10)


    splitAudio(argsMain)
    recogAudio(argsCNN)
    sumAudio(argsMain.data_dir)
    cleanAudio()

    print("####################")
    print("time :", time.time() - start)  # 현재시각 - 시작시간 = 실행 시간
    print("####################")