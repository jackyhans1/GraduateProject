import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

# GPU 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 데이터셋 클래스
class SegDataset(Dataset):
    def __init__(self, mfcc_list, emotion_list):
        self.mfcc = mfcc_list
        self.emotion = emotion_list
        print(f"Dataset loaded with {len(self.mfcc)} samples.")

    def __getitem__(self, idx):
        mfcc = self.mfcc[idx].to(device)  # 데이터를 GPU로 이동
        emotion = torch.tensor(self.emotion[idx], dtype=torch.long).to(device)  # 레이블도 GPU로 이동

        return mfcc, emotion 
    
    def __len__(self):
        return len(self.mfcc)


# MFCC와 레이블 불러오기 함수 정의
def load_mfcc_and_labels(mfcc_root):
    file_paths = []

    # 루트 디렉토리 존재 여부 확인
    if not os.path.exists(mfcc_root):
        raise FileNotFoundError(f"Root directory '{mfcc_root}' does not exist.")

    # 감정 디렉토리 탐색 및 MFCC 파일 수집
    for emotion_dir in os.listdir(mfcc_root):
        emotion_path = os.path.join(mfcc_root, emotion_dir)
        if os.path.isdir(emotion_path):
            for file in os.listdir(emotion_path):
                file_path = os.path.join(emotion_path, file)
                if os.path.isfile(file_path) and file.endswith('.pt'):  # .pt 파일만 수집
                    file_paths.append((file_path, emotion_dir))

    # 파일이 존재하지 않을 경우 예외 처리
    if len(file_paths) == 0:
        raise ValueError(f"No files found in directory '{mfcc_root}'.")

    mfcc_list = []
    emotion_list = []

    # 감정 매핑: 각 디렉토리 이름을 숫자 레이블로 매핑
    emotion_mapping = {
        'happy': 0, 'neutral': 0, 'lovely': 0,  # Not Stressed: 0
        'angry': 1, 'surprise': 1, 'sad': 1, 'fear': 1  # Stressed: 1
    }

    # 파일마다 MFCC 특징 및 감정 레이블 부여
    for file_path, emotion_dir in tqdm(file_paths, desc="Loading MFCC files", unit="file"):
        try:
            # MFCC 텐서 불러오기
            mfcc = torch.load(file_path, weights_only=True)  # .pt 파일로 저장된 텐서를 불러옴
        except Exception as e:
            print(f"Error loading file {file_path}: {e}")
            continue

        # 감정 레이블 매핑
        if emotion_dir in emotion_mapping:
            emotion = emotion_mapping[emotion_dir]
        else:
            print(f"Skipping file {file_path}, no matching emotion label for directory '{emotion_dir}'.")
            continue

        # mfcc 리스트와 감정 리스트에 데이터 추가
        mfcc_list.append(mfcc)
        emotion_list.append(emotion)

    return mfcc_list, emotion_list

# KFold 데이터를 생성하는 함수
def get_kfold_data(n_splits, random_state, mfcc_directory):
    # 미리 생성된 MFCC 파일들을 로딩
    mfcc_list, emotion_list = load_mfcc_and_labels(mfcc_directory)
    
    # Stratified K-Fold 설정
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    fold_data = []
    
    # 각 fold에 대해 데이터를 나누기
    for train_indices, val_indices in skf.split(mfcc_list, emotion_list):
        mfcc_train = [mfcc_list[i] for i in train_indices]
        mfcc_val = [mfcc_list[i] for i in val_indices]
        emotion_train = [emotion_list[i] for i in train_indices]
        emotion_val = [emotion_list[i] for i in val_indices]
        
        fold_data.append((mfcc_train, mfcc_val, emotion_train, emotion_val))
    
    return fold_data

# 기본 설정
n_splits = 5
random_state = 42
mfcc_directory = '/workspace/dataset/free_talking_datasets/mfcc_split_voice_train'
batch_size = 16

if __name__ == "__main__":
    try:
        # K-Fold 데이터를 준비
        k_folds_data = get_kfold_data(n_splits=n_splits, random_state=random_state, mfcc_directory=mfcc_directory)
        print("K-Fold 데이터 준비 완료")
        # 각 fold를 순회하며 데이터를 확인
        for fold_idx, (train_mfcc, val_mfcc, train_emotion, val_emotion) in enumerate(k_folds_data):
            print(f"Fold {fold_idx + 1} | Train size: {len(train_mfcc)}, Validation size: {len(val_mfcc)}")
            
            # SegDataset 인스턴스를 생성하여 DataLoader로 감쌈
            train_dataset = SegDataset(train_mfcc, train_emotion)
            val_dataset = SegDataset(val_mfcc, val_emotion)

            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0) 
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

            # Dataset 길이 출력
            print(f"Train Dataset Length: {len(train_dataset)}, Validation Dataset Length: {len(val_dataset)}")

            # 각 감정 클래스의 수를 계산하고 출력
            emotion_0_count = train_emotion.count(0)
            emotion_1_count = train_emotion.count(1)
            print(f"Fold {fold_idx + 1}: Emotion 0 count (Not Stressed): {emotion_0_count}, Emotion 1 count (Stressed): {emotion_1_count}")

            # 샘플 데이터를 로딩하고, 그 모양과 레이블을 출력하여 확인
            sample_mfcc, sample_emotion = next(iter(train_loader))
            print(f"Sample MFCC batch shape: {sample_mfcc.shape}")
            print(f"Sample Emotion batch: {sample_emotion}")

    except Exception as e:
        print(f"An error occurred: {e}")


"""
K-Fold 데이터 준비 완료
Fold 1 | Train size: 210558, Validation size: 52640
Dataset loaded with 210558 samples.
Dataset loaded with 52640 samples.
Train Dataset Length: 210558, Validation Dataset Length: 52640
Fold 1: Emotion 0 count (Not Stressed): 158388, Emotion 1 count (Stressed): 52170
Sample MFCC batch shape: torch.Size([16, 2, 16, 626])
Sample Emotion batch: tensor([1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0], device='cuda:0')
"""
