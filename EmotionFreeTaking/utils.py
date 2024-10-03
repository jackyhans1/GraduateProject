import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import numpy as np

import torch
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
import torchaudio
import torchaudio.transforms as T

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

def get_mfcc(path, n_mfcc):
    # 파일 로드 (torchaudio는 바로 Tensor 형태로 로드)
    waveform, sr = torchaudio.load(path)
    
    # GPU로 데이터 이동
    waveform = waveform.to(device)

    # Normalize the length of the audio (패딩 또는 자르기 적용)
    VOICE_LEN = 160000  # 10초 기준
    if waveform.shape[1] < VOICE_LEN:
        pad_len = VOICE_LEN - waveform.shape[1]
        waveform = torch.nn.functional.pad(waveform, (0, pad_len))
    else:
        waveform = waveform[:, :VOICE_LEN]

    # Frame length에 맞는 N_FFT 값 구하기
    N_FFT = getNearestLen(0.25, sr)

    # MFCC 변환 (GPU에서 수행)
    mfcc_transform = T.MFCC(
        sample_rate=sr,
        n_mfcc=n_mfcc,
        melkwargs={
            'n_fft': N_FFT,
            'hop_length': int(N_FFT / 4),
            'f_max': sr // 2
        }
    ).to(device)

    # MFCC 데이터 계산 (GPU에서)
    mfcc_data = mfcc_transform(waveform)
    
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
    # y_pred가 확률일 경우 0.5 기준으로 이진 분류로 변환
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

def plot_fold_performance(train_losses, val_losses, val_accuracies, fold):
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title(f'Fold {fold + 1} Losses')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.title(f'Fold {fold + 1} Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()