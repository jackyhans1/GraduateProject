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

# CUDA 장치 설정
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# 다중 프로세싱 시작 방식 설정
try:
    mp.set_start_method('spawn', force=True)
except RuntimeError:
    print("Multiprocessing start method already set. Please restart the runtime.")

# 사용자 정의 모듈 임포트
try:
    # 세션에 저장된 모듈 제거 후 재임포트
    if "models_dh" in sys.modules:
        importlib.reload(sys.modules["models_dh"])
    if "Segdataset" in sys.modules:
        importlib.reload(sys.modules["Segdataset"])
    if "utils" in sys.modules:
        importlib.reload(sys.modules["utils"])

    from models_dh import *
    from Segdataset import SegDataset, get_kfold_data
    from utils import plot_fold_performance2, get_mfcc
except ImportError as e:
    sys.exit(f"Failed to import required modules: {e}")

def main():
    # CUDA 사용 가능 여부 확인
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(device)

    # 설정 클래스 정의
    class Args:
        def __init__(self):
            self.data_root = '/workspace/dataset/CREMA-D/train'
            self.save_root = '/workspace/UndergraduateResearchAssistant/GraduateProject/code/CREMA-D/checkpoints_dh'
            self.result_root = '/workspace/UndergraduateResearchAssistant/GraduateProject/code/CREMA-D/result_dh'
            self.epoch = 100
            self.lr = 5e-4
            self.batch_size = 1028
            self.num_workers = 16
            self.random_seed = 1
            self.n_mfcc = 20
            self.n_splits = 5
            self.model_type = 'ResNetCNN'

    # 설정 인스턴스 생성
    opt = Args()

    # 모델 초기화
    if opt.model_type == 'CNN':
        model = CNN().to(device)
    elif opt.model_type == 'ResNetCNN':
        model = ResNetCNN().to(device)
    elif opt.model_type == 'RNN':
        model = RNN(n_mfcc=opt.n_mfcc).to(device)
    elif opt.model_type == 'Transformer':
        model = Transformer(n_mfcc=opt.n_mfcc).to(device)
    else:
        raise ValueError(f"Invalid model type specified: {opt.model_type}")

    print(model)

    # 데이터셋 로딩
    fold_data = get_kfold_data(root=opt.data_root, n_mfcc=opt.n_mfcc)

    # 학습 및 검증 루프 시작
    for fold, (train_mfcc, val_mfcc, train_emotion, val_emotion) in enumerate(fold_data):
        print(f"Fold {fold + 1}/{opt.n_splits}")

        # 모델 초기화
        model = ResNetCNN().to(device)

        # 학습/검증 데이터셋 및 로더 생성
        train_set = SegDataset(train_mfcc, train_emotion)
        val_set = SegDataset(val_mfcc, val_emotion)

        train_loader = DataLoader(train_set, batch_size=opt.batch_size, shuffle=True, drop_last=True, num_workers=opt.num_workers)
        val_loader = DataLoader(val_set, batch_size=opt.batch_size, shuffle=False, drop_last=False, num_workers=opt.num_workers)

        # 각 fold에 따라 pos_weight 동적으로 설정
        num_pos = sum(em == 1 for em in train_emotion)
        num_neg = sum(em == 0 for em in train_emotion)
        pos_weight = torch.tensor([num_neg / num_pos]).to(device) if num_pos > 0 else torch.tensor([1.0]).to(device)

        # 손실 함수 및 옵티마이저 선언
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)

        # 현재 폴드 값들 저장
        train_losses = []
        val_losses = []
        val_accuracies = []

        best_val_loss = float('inf')
        best_epoch = 0

        # 에포크 루프
        for epo in range(opt.epoch):
            # 모델 학습 모드 전환
            model.train()
            train_loss = 0

            # 학습 루프
            for mfcc, emotion in train_loader:
                # 데이터를 GPU로 옮기기
                mfcc, emotion = mfcc.to(device), emotion.to(device)

                # Optimizer 초기화
                optimizer.zero_grad()

                # 모델 예측 및 손실 계산
                output_emotion = model(mfcc)
                loss = criterion(output_emotion[:, 0].squeeze(), emotion.float())

                # 역전파 및 옵티마이저 스텝
                loss.backward()
                optimizer.step()

                # 손실 기록
                train_loss += loss.item()

                # 메모리 최적화
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

                    # 모델 예측 및 손실 계산
                    output_emotion = model(mfcc)
                    loss = criterion(output_emotion[:, 0].squeeze(), emotion.float())
                    val_loss += loss.item()

                    # 예측 값과 실제 값 비교하여 정확도 계산
                    predicted_emotion = (torch.sigmoid(output_emotion[:, 0]) > 0.5).float()
                    correct_emotion += (predicted_emotion == emotion).sum().item()
                    total += emotion.size(0)

                    # 메모리 최적화
                    del mfcc, emotion, output_emotion
                    torch.cuda.empty_cache()

            # 정확도 계산
            val_accuracy = correct_emotion / total if total > 0 else 0.0

            # 결과 출력
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

        # 각 폴드의 학습 손실, 검증 손실 및 정확도 시각화 및 저장
        result_path = os.path.join(opt.result_root, f'fold_{fold + 1}_performance.png')
        plot_fold_performance2(train_losses, val_losses, val_accuracies, fold, result_path)

if __name__ == "__main__":
    main()
