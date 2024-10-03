import torch
torch.cuda.empty_cache()

# 필요한 라이브러리 임포트
import os
import sys
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.multiprocessing as mp
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader
import importlib 

# CUDA 장치 설정 (특정 GPU 사용)
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# 다중 프로세싱 시작 방식 설정 (DataLoader 사용 시)
try:
    mp.set_start_method('spawn', force=True)
except RuntimeError:
    print("Multiprocessing start method already set. Please restart the runtime.")

# 사용자 정의 모듈 임포트 (기존 임포트된 모듈 제거 및 새로 임포트)
try:
    # 세션에 저장된 모듈 제거
    if "models" in sys.modules:
        importlib.reload(sys.modules["models"])
    if "Segdataset" in sys.modules:
        importlib.reload(sys.modules["Segdataset"])
    if "utils" in sys.modules:
        importlib.reload(sys.modules["utils"])

    # 사용자 정의 모듈 임포트
    from models import *
    from Segdataset import SegDataset, load_mfcc_and_labels  # 데이터셋 클래스 및 파일 읽기 함수 가져오기
    from utils import plot_fold_performance  # 모델 성능 시각화 함수 가져오기

except ImportError as e:
    sys.exit(f"Failed to import required modules: {e}")


if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')


class Args:
    def __init__(self):
        # 데이터 경로 설정
        self.data_root = '/workspace/dataset/free_talking_datasets/mfcc_split_voice_train'  # MFCC 파일이 저장된 루트 디렉토리
        self.save_root = 'checkpoints/CNN'  # 모델 체크포인트를 저장할 경로

        # 학습 설정
        self.epoch = 50  # 학습 에포크 수
        self.lr = 3e-5  # 학습률
        self.batch_size = 16  # 배치 크기
        self.num_workers = 1  # DataLoader에서 사용할 병렬 워커 수

        # 랜덤 시드 및 특성 설정
        self.random_seed = 1  # 재현 가능성을 위한 랜덤 시드
        self.n_mfcc = 16  # MFCC 특성 차원 수

        # K-Fold 교차 검증 설정
        self.n_splits = 5  # 교차 검증을 위한 K-fold 개수

        # 모델 설정 (CNN, RNN, Transformer 중 선택)
        self.model_type = 'CNN'  # 사용할 모델 타입 ('CNN', 'RNN', 'Transformer' 중 하나)

# 설정 인스턴스 생성
opt = Args()


# 모델 선택 및 초기화
if opt.model_type == 'CNN':
    model = CNN().to(device)
elif opt.model_type == 'RNN':
    model = RNN(n_mfcc=opt.n_mfcc).to(device)
elif opt.model_type == 'Transformer':
    model = Transformer(n_mfcc=opt.n_mfcc).to(device)
else:
    raise ValueError(f"Invalid model type specified: {opt.model_type}")

print(model)

# MFCC 특징과 감정 레이블 로드
mfcc_list, emotion_list = load_mfcc_and_labels(opt.data_root) # GPU에 데이터 적제 되어 있음

# K-Fold 설정
kf = KFold(n_splits=opt.n_splits, shuffle=True, random_state=opt.random_seed)

# 학습 및 검증 루프 시작
for fold, (train_idx, val_idx) in enumerate(kf.split(mfcc_list)):
    print(f"Fold {fold + 1}/{opt.n_splits}")

    # 모델 초기화
    if opt.model_type == 'CNN':
        model = CNN().to(device)
    elif opt.model_type == 'RNN':
        model = RNN(n_mfcc=opt.n_mfcc).to(device)
    elif opt.model_type == 'Transformer':
        model = Transformer(n_mfcc=opt.n_mfcc).to(device)
    else:
        raise ValueError(f"Invalid model type: {opt.model_type}")

    # 학습/검증 데이터로 쪼개기
    train_mfcc = [mfcc_list[i] for i in train_idx]
    train_emotion = [emotion_list[i] for i in train_idx]
    val_mfcc = [mfcc_list[i] for i in val_idx]
    val_emotion = [emotion_list[i] for i in val_idx]

    train_set = SegDataset(train_mfcc, train_emotion)
    val_set = SegDataset(val_mfcc, val_emotion)

    # 데이터 로더 생성 (각 배치 단위로 데이터를 GPU에 올리기)
    train_loader = DataLoader(train_set, batch_size=opt.batch_size, shuffle=True, drop_last=True, num_workers=opt.num_workers)
    val_loader = DataLoader(val_set, batch_size=opt.batch_size, shuffle=False, drop_last=True, num_workers=opt.num_workers)

    # 각 fold에 따라 pos_weight 동적으로 설정
    num_pos = sum(em == 1 for em in train_emotion)
    num_neg = sum(em == 0 for em in train_emotion)
    pos_weight = torch.tensor([num_neg / num_pos]).to(device) if num_pos > 0 else torch.tensor([1.0]).to(device)

    # 손실 함수 및 옵티마이저 선언
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight).to(device)
    optimizer = optim.Adam(model.parameters(), lr=opt.lr)

    # 현재 폴드 값들 저장
    train_losses = []
    val_losses = []
    val_accuracies = []

    best_val_loss = float('inf')
    best_epoch = 0

    # 에포크 루프
    for epo in range(opt.epoch):
        model.train()
        train_loss = 0

        # 학습 루프 (각 배치 단위로 데이터를 GPU에 올리기)
        for mfcc, emotion in train_loader:
            # 데이터를 GPU로 옮기기
            mfcc, emotion = mfcc.to(device), emotion.to(device)
            optimizer.zero_grad()
            output_emotion = model(mfcc)

            emotion = emotion.float()
            loss = criterion(output_emotion[:, 0].squeeze(), emotion)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            # 사용 후 메모리에서 해제
            del mfcc, emotion, output_emotion
            torch.cuda.empty_cache()

        # 검증 루프
        model.eval()
        val_loss = 0
        correct_emotion = 0
        total = 0
        with torch.no_grad():
            for mfcc, emotion in val_loader:
                # 데이터를 GPU로 옮기기
                mfcc, emotion = mfcc.to(device), emotion.to(device)
                output_emotion = model(mfcc)
                emotion = emotion.float()

                loss = criterion(output_emotion[:, 0].squeeze(), emotion)
                val_loss += loss.item()

                predicted_emotion = (torch.sigmoid(output_emotion[:, 0]) > 0.56).float()
                correct_emotion += (predicted_emotion == emotion).sum().item()
                total += emotion.size(0)

                # 사용 후 메모리에서 해제
                del mfcc, emotion, output_emotion
                torch.cuda.empty_cache()

        val_accuracy = correct_emotion / total
        print(f"Fold {fold + 1} | Epoch {epo} | Train Loss: {train_loss / len(train_loader)} | Val Loss: {val_loss / len(val_loader)} | Val Accuracy: {val_accuracy}")

        # 모델 저장
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epo
            torch.save(model.state_dict(), os.path.join(opt.save_root, f'fold_{fold + 1}_best_epoch.pth'))

        # 학습 및 검증 손실 및 정확도 기록
        train_losses.append(train_loss / len(train_loader))
        val_losses.append(val_loss / len(val_loader))
        val_accuracies.append(val_accuracy)

    print(f"Fold {fold + 1} best epoch: {best_epoch}")

    # 각 폴드의 학습 손실, 검증 손실 및 정확도 시각화
    plot_fold_performance(train_losses, val_losses, val_accuracies, fold , opt.save_root)
    