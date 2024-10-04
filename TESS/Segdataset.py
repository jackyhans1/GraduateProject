import os
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import StratifiedKFold
from utils import get_mfcc

# Define emotion classes
emotions = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'ps', 'sad']

def read_train_file_list(root='datasets/train', n_mfcc=16):
    data_path = root
    file_paths = []

    # Root directory가 존재하는지 확인
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Root directory '{data_path}' does not exist.")

    # train_data 디렉토리 내 파일을 탐색
    for file in os.listdir(data_path):
        if file == '.DS_Store':  # 무시할 파일
            continue

        file_path = os.path.join(data_path, file)
        if os.path.isfile(file_path) and file.endswith('.wav'):
            file_paths.append(file_path)
    if len(file_paths) == 0:
        raise ValueError(f"No files found in directory '{data_path}'.")

    mfcc_list = []
    emotion_list = []

    for file_path in file_paths:
        try:
            mfcc = get_mfcc(file_path, n_mfcc)  # get_mfcc 함수는 기존에 정의된 것으로 가정
        except Exception as e:
            print(f"Error processing file {file_path}: {e}")
            continue

        # 파일 이름에 따라 감정을 분류
        if 'happy' in file_path or 'neutral' in file_path or 'ps' in file_path:
            emotion = 0  # NotStressed
        elif 'angry' in file_path or 'disgust' in file_path or 'sad' in file_path or 'fear' in file_path:
            emotion = 1  # Stressed
        else:
            print(f"Skipping file {file_path}, no matching emotion label.")
            continue

        mfcc_list.append(mfcc)
        emotion_list.append(emotion)

    if len(mfcc_list) == 0 or len(emotion_list) == 0:
        raise ValueError("No valid data found after processing files.")

    return mfcc_list, emotion_list

def read_test_file_list(root='datasets/test', n_mfcc=16):
    data_path = root 
    file_paths = []

    # Root directory가 존재하는지 확인
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Root directory '{data_path}' does not exist.")

    # test_data 디렉토리 내 파일을 탐색
    for file in os.listdir(data_path):
        if file == '.DS_Store':  # 무시할 파일
            continue

        file_path = os.path.join(data_path, file)
        if os.path.isfile(file_path) and file.endswith('.wav'):
            file_paths.append(file_path)

    if len(file_paths) == 0:
        raise ValueError(f"No files found in directory '{data_path}'.")

    mfcc_list = []
    emotion_list = []

    for file_path in file_paths:
        try:
            mfcc = get_mfcc(file_path, n_mfcc)  # get_mfcc 함수는 기존에 정의된 것으로 가정
        except Exception as e:
            print(f"Error processing file {file_path}: {e}")
            continue

        # 파일 이름에 따라 감정 분류
        if 'happy' in file_path or 'neutral' in file_path or 'ps' in file_path:
            emotion = 0  # NotStressed
        elif 'angry' in file_path or 'disgust' in file_path or 'sad' in file_path or 'fear' in file_path:
            emotion = 1  # Stressed
        else:
            print(f"Skipping file {file_path}, no matching emotion label.")
            continue

        mfcc_list.append(mfcc)
        emotion_list.append(emotion)

    if len(mfcc_list) == 0 or len(emotion_list) == 0:
        raise ValueError("No valid data found after processing files.")

    return mfcc_list, emotion_list

def get_kfold_data(k_folds=5, random_state=1, root='datasets/train', n_mfcc=16):
    # Read train dataset only
    mfcc_list, emotion_list = read_train_file_list(root=root, n_mfcc=n_mfcc)
    
    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=random_state)
    
    fold_data = []
    
    # K-fold split
    for train_index, val_index in skf.split(mfcc_list, emotion_list):
        mfcc_train = [mfcc_list[i] for i in train_index]
        mfcc_val = [mfcc_list[i] for i in val_index]
        emotion_train = [emotion_list[i] for i in train_index]
        emotion_val = [emotion_list[i] for i in val_index]
        
        fold_data.append((mfcc_train, mfcc_val, emotion_train, emotion_val))
    return fold_data


class SegDataset(Dataset):
    def __init__(self, mfcc_list, emotion_list):
        self.mfcc = mfcc_list
        self.emotion = emotion_list
        print(f"Dataset loaded with {len(self.mfcc)} samples.")

    def __getitem__(self, idx):
        mfcc = self.mfcc[idx]
        emotion = self.emotion[idx]

        # Convert to Torch tensor and adjust dimensions
        mfcc = torch.from_numpy(np.array(mfcc)).type(torch.FloatTensor).transpose(1, 0)
        emotion = torch.tensor(emotion, dtype=torch.long)

        return mfcc, emotion
    
    def __len__(self):
        return len(self.mfcc)

if __name__ == "__main__":
    try:
        k_folds_data = get_kfold_data(k_folds=5, random_state=1, root='datasets/train', n_mfcc=16)
        
        for fold_idx, (train_mfcc, val_mfcc, train_emotion, val_emotion) in enumerate(k_folds_data):
            print(f"Fold {fold_idx + 1} | Train size: {len(train_mfcc)}, Val size: {len(val_mfcc)}")
            
            # Load the training dataset for the current fold
            train_dataset = SegDataset(train_mfcc, train_emotion)
            val_dataset = SegDataset(val_mfcc, val_emotion)

            print(f"Train Dataset Length: {len(train_dataset)}, Val Dataset Length: {len(val_dataset)}")
            
            emotion_0_count = val_emotion.count(0)
            emotion_1_count = val_emotion.count(1)
            print(f"Fold {fold_idx + 1}: Emotion 0 count: {emotion_0_count}, Emotion 1 count: {emotion_1_count}")
            
            sample_mfcc, sample_emotion = train_dataset[0]
            print("Sample MFCC shape:", sample_mfcc.shape)
            print("Sample Emotion Label:", sample_emotion)
    except Exception as e:
        print(f"An error occurred: {e}")
