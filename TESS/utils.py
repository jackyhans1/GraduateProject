import numpy as np
import librosa
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score,confusion_matrix,precision_score,recall_score,f1_score

def normalizeVoiceLen(y,normalizedLen):
    nframes=len(y)
    y = np.reshape(y,[nframes,1]).T

    if(nframes<normalizedLen):
        res=normalizedLen-nframes
        res_data=np.zeros([1,res],dtype=np.float32)
        y = np.reshape(y,[nframes,1]).T
        y=np.c_[y,res_data]
    else:
        y=y[:,0:normalizedLen]
    return y[0]

def getNearestLen(framelength,sr):
    framesize = framelength*sr

    nfftdict = {}
    lists = [32,64,128,256,512,1024]
    for i in lists:
        nfftdict[i] = abs(framesize - i)
    sortlist = sorted(nfftdict.items(), key=lambda x: x[1])
    framesize = int(sortlist[0][0])
    return framesize

def get_mfcc(path,n_mfcc):
    y,sr = librosa.load(path,sr=16000)
    VOICE_LEN=32000

    N_FFT=getNearestLen(0.25,sr)

    y=normalizeVoiceLen(y,VOICE_LEN)

    mfcc_data=librosa.feature.mfcc(y=y, sr=sr,n_mfcc=n_mfcc,n_fft=N_FFT,hop_length=int(N_FFT/4),fmax = sr//2)
    return mfcc_data

def plot_confusion_matrix(y_true, y_pred, labels):
    # y_true와 y_pred가 1차원인지 확인
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()

    plt.figure()
    num = len(labels)
    
    # confusion matrix 생성
    C = confusion_matrix(y_true=y_true, y_pred=y_pred, labels=range(num))
    
    # confusion matrix 시각화
    plt.matshow(C, cmap=plt.cm.Reds)
    
    # matrix 값 추가
    for i in range(C.shape[0]):
        for j in range(C.shape[1]):
            plt.text(j, i, str(C[i, j]), ha='center', va='center', color='black')
    
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
    # X축과 Y축에 라벨 추가
    plt.xticks(range(num), labels=labels)
    plt.yticks(range(num), labels=labels)

    # plt.show()  # confusion matrix 시각화


def get_evaluation(y_true, y_pred):
    # y_pred가 확률일 경우 0.56 기준으로 이진 분류로 변환
    y_pred = np.array(y_pred)
    print(y_pred.shape)
    y_true = np.array(y_true).flatten()  # y_true를 1차원으로 변환
    y_pred_binary = (y_pred > 0.56).astype(int)  # 확률 값을 0과 1로 변환

    print(y_true.shape)  # 확인
    print(y_pred_binary.shape)  # 확인

    # 정확도, 정밀도, 재현율, f1 스코어 계산
    acc = accuracy_score(y_true, y_pred_binary)
    precision = precision_score(y_true, y_pred_binary, average='binary')
    recall = recall_score(y_true, y_pred_binary, average='binary')
    f1 = f1_score(y_true, y_pred_binary, average='binary')

    # 소수점 3자리로 반올림
    precision = np.around(precision, 3)
    recall = np.around(recall, 3)
    f1 = np.around(f1, 3)

    return acc, precision, recall, f1