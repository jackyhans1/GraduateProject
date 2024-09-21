import torch
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(p=0.3),  # 드롭아웃 추가 (Conv2d 이후)
            
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout(p=0.3),  # 드롭아웃 추가 (Conv2d 이후)

            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.AdaptiveAvgPool2d(output_size=(1, 1))
        )
        
        self.fc1 = nn.Linear(128, 2)
        self.dropout_fc = nn.Dropout(p=0.5)  # Fully Connected layer 앞 드롭아웃 추가
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = x.unsqueeze(dim=1)  # 차원 확장
        x = self.cnn(x).squeeze()
        
        x = self.dropout_fc(x)  # FC 레이어 앞 드롭아웃
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
        emotion = self.sigmoid(emotion)  # 이진 분류이므로 sigmoid 사용
        return emotion


class Transformer(nn.Module):
    def __init__(self, n_mfcc=16):
        super().__init__()

        self.encoder_layer = nn.TransformerEncoderLayer(d_model=n_mfcc, nhead=4, batch_first=True, dim_feedforward=512)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=4)
        self.avg = nn.AdaptiveAvgPool1d(output_size=1)
        self.fc1 = nn.Linear(n_mfcc, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.transformer_encoder(x).transpose(2, 1)
        x = self.avg(x).squeeze()
        emotion = self.fc1(x)
        emotion = self.sigmoid(emotion)  # 이진 분류를 위한 sigmoid 함수 적용
        return emotion

