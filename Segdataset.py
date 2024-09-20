import os
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
import librosa  # librosa 추가

# 감정 레이블 정의
emotions = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'ps', 'sad']

# MFCC 추출 함수 수정
def get_mfcc(file_path, n_mfcc=16, sr=16000, n_mels=32, fmax=None):
    """
    MFCC 특징을 추출하는 함수.
    
    Args:
        file_path (str): 오디오 파일 경로
        n_mfcc (int): MFCC 차원 수
        sr (int): 샘플링 레이트
        n_mels (int): 멜 필터 수
        fmax (float): 최대 주파수 (None일 경우 sr/2 사용)
    
    Returns:
        mfcc (ndarray): 추출된 MFCC 특징
    """
    y, sr = librosa.load(file_path, sr=sr)  # 샘플링 레이트 설정
    if fmax is None:
        fmax = sr / 2  # 최대 주파수 기본값 설정
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, fmax=fmax, n_mels=n_mels)  # n_mels 및 fmax 조정
    return mfcc

def read_file_list(root='datasets', n_mfcc=16, random_state=1, test_size=0.25):
    """
    파일 목록을 읽고 학습/검증 데이터셋으로 분리.

    Args:
        root (str): 데이터셋 루트 디렉터리
        n_mfcc (int): MFCC 차원 수
        random_state (int): 랜덤 시드 값
        test_size (float): 검증 데이터 비율

    Returns:
        train_data (tuple): 학습 데이터셋 (MFCC, 감정 레이블)
        val_data (tuple): 검증 데이터셋 (MFCC, 감정 레이블)
    """
    file_path_all = []
    
    # 디렉터리에서 파일 경로 수집 (디렉터리 내의 파일만)
    for dir_name in os.listdir(root):
        dir_root = os.path.join(root, dir_name)
        
        if not os.path.isdir(dir_root):
            continue
        
        for file_name in os.listdir(dir_root):
            file_path = os.path.join(dir_root, file_name)
            if os.path.isfile(file_path):
                file_path_all.append(file_path)

    # 파일을 학습용과 검증용으로 분리 (여기서 중복 없이 분리)
    train_files, val_files = train_test_split(file_path_all, test_size=test_size, random_state=random_state)

    def process_files(file_paths):
        mfcc_list, emotion_list = [], []
        stressed_emotions = ['angry', 'disgust', 'fear', 'sad']

        for file_path in file_paths:
            mfcc = get_mfcc(file_path, n_mfcc)
                        
            # 감정 레이블을 이진 스트레스 레이블로 변환
            emotion = 1 if any(emotion_label in file_path for emotion_label in stressed_emotions) else 0
            mfcc_list.append(mfcc)
            emotion_list.append(emotion)

        return mfcc_list, emotion_list
    
    # 학습 데이터셋과 검증 데이터셋을 처리
    train_data = process_files(train_files)
    val_data = process_files(val_files)

    return train_data, val_data

class SegDataset(Dataset):
    def __init__(self, data):
        """
        사용자 정의 Dataset 클래스.
        
        Args:
            data (tuple): (MFCC 데이터 리스트, 감정 레이블 리스트)
        """
        self.mfcc, self.emotion = data
        print(f'Read {len(self.mfcc)} examples.')

    def __getitem__(self, idx):
        mfcc = torch.from_numpy(np.array(self.mfcc[idx])).type(torch.FloatTensor).transpose(1, 0)
        emotion = torch.tensor(self.emotion[idx], dtype=torch.float32)
        return mfcc, emotion

    def __len__(self):
        return len(self.mfcc)

