import os
import cv2
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from sklearn.model_selection import train_test_split
from utils import get_mfcc
emotions=['angry','disgust','fear','happy','neutral','ps','sad']

def read_file_list(root=r'datasets', type='train', n_mfcc=16, random_state=1, test_size=0.25):
    root = os.path.join(root, type)
    file_path_all = []
    
    # 디렉토리에서 파일 경로 수집
    for dir_name in os.listdir(root):
        dir_root = os.path.join(root, dir_name)
        
        # dir_root가 디렉토리인지 확인
        if not os.path.isdir(dir_root):
            continue
        
        for file in os.listdir(dir_root):
            # .DS_Store 파일을 무시
            if file == '.DS_Store':
                continue
            
            file_path = os.path.join(dir_root, file)
            file_path_all.append(file_path)
    
    mfcc_list = []
    emotion_list = []
    
    # 감정 상태에 따라 mfcc와 emotion 리스트 구성
    for file_path in file_path_all:
        mfcc = get_mfcc(file_path, n_mfcc)

        if 'happy' in file_path or 'neutral' in file_path or 'ps' in file_path:
            emotion = 0  # NotStressed
        elif 'angry' in file_path or 'disgust' in file_path or 'sad' in file_path or 'fear' in file_path:
            emotion = 1  # Stressed
        
        mfcc_list.append(mfcc)
        emotion_list.append(emotion)
    
    mfcc_train, mfcc_val, emotion_train, emotion_val = train_test_split(mfcc_list, emotion_list, test_size=test_size, random_state=random_state)

    if type == 'train':
        return mfcc_train, emotion_train
    elif type == 'val':
        return mfcc_val, emotion_val


class SegDataset(torch.utils.data.Dataset):
    def __init__(self, root=r'datasets', type='train', n_mfcc=16, random_state=1, test_size=0.25):

        mfcc, emotion= read_file_list(root=root, type=type, n_mfcc=n_mfcc, random_state=random_state, test_size=test_size)

        self.mfcc = mfcc
        self.emotion = emotion

        print('Read ' + str(len(self.mfcc)) + ' valid examples')


    def __getitem__(self, idx):
        mfcc = self.mfcc[idx]
        emotion = self.emotion[idx]

        mfcc = torch.from_numpy(np.array(mfcc)).type(torch.FloatTensor).transpose(1,0)
        emotion = torch.from_numpy(np.array(emotion)).long()



        return mfcc, emotion  # float32 tensor, uint8 tensor
    
    def __len__(self):
        return len(self.mfcc)

if __name__ == "__main__":

    voc_train = SegDataset()
    print(type(voc_train))#<class '__main__.VOCSegDataset'>
    print(len(voc_train))
    img, label,_ = voc_train[11]
    # img=np.transpose(np.array(img, np.float64), [1, 2, 0])
    print(img)
    print(label)
    # plt.imshow(img)
    # plt.show()
    print(type(img), type(label))
    print(img.shape, label.shape, _.shape)
    print(label,_)

