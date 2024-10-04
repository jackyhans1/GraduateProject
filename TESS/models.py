import torch
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=16, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.AdaptiveAvgPool1d(output_size=1)
        )
        self.fc1 = nn.Linear(128, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x.permute(0, 2, 1)  # [배치 크기, 16, 126]
        x = self.cnn(x)
        x = x.squeeze()
        emotion = self.fc1(x)
        emotion = self.sigmoid(emotion)
        return emotion

    
class RNN(nn.Module):
    def __init__(self, n_mfcc=16):
        super().__init__()

        self.LSTM = nn.LSTM(n_mfcc, 40, 4, batch_first=True)
        self.avg = nn.AdaptiveAvgPool1d(output_size=1)
        self.fc1 = nn.Linear(40, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.LSTM(x)[0].transpose(2, 1)
        x = self.avg(x).squeeze()
        emotion = self.fc1(x)
        emotion = self.sigmoid(emotion)
        return emotion

class Transformer(nn.Module):
    def __init__(self, n_mfcc=16):
        super().__init__()

        self.encoder_layer = nn.TransformerEncoderLayer(d_model=n_mfcc, nhead=4, batch_first=True, dim_feedforward=512)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=4)
        self.avg = nn.AdaptiveAvgPool1d(output_size=1)
        self.fc1 = nn.Linear(n_mfcc, 1)  # 이진 분류이므로 출력 차원을 1로 설정
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.transformer_encoder(x).transpose(2, 1)
        x = self.avg(x).squeeze()
        emotion = self.fc1(x)
        emotion = self.sigmoid(emotion)  # 이진 분류에 맞게 Sigmoid 사용
        return emotion