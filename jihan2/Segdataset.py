import os
from torch.utils.data import Dataset
from PIL import Image
import torch
from torchvision import transforms

# 감정 클래스
emotions = ['ANG', 'DIS', 'FEA', 'HAP', 'NEU', 'SAD']  # 논문에서 사용한 6가지 감정
emotion_to_idx = {emotion: idx for idx, emotion in enumerate(emotions)}

# 사용자 정의 데이터셋 클래스
class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.img_list = []
        self.trg_list = []
        
        # 파일명에서 감정 레이블 추출
        for fname in os.listdir(root_dir):
            if fname.endswith('.png'):
                emotion = fname.split('_')[2]  # 파일명에서 감정 부분 추출
                if emotion in emotion_to_idx:
                    self.img_list.append(os.path.join(root_dir, fname))
                    self.trg_list.append(emotion_to_idx[emotion])
    
    def __len__(self):
        return len(self.img_list)
    
    def __getitem__(self, idx):
        img_path = self.img_list[idx]
        img = Image.open(img_path).convert('RGB')  # RGB 이미지로 로드
        if self.transform:
            img = self.transform(img)
        trg = torch.tensor(self.trg_list[idx], dtype=torch.long)
        return img, trg

# 데이터 전처리 설정
def get_transform():
    return transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])
