import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from models import Teacher, Student
from Segdataset import CustomDataset, get_transform

# 환경 변수 설정
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# 체크포인트 경로 설정
CHECKPOINT_DIR = '/workspace/UndergraduateResearchAssistant/GraduateProject/code/VIT/jihan2/checkpoints/ViT'
TEACHER_DIR = os.path.join(CHECKPOINT_DIR, 'teacher')
STUDENT_DIR = os.path.join(CHECKPOINT_DIR, 'student')

# 데이터셋 경로
train_data_path = '/workspace/dataset/CREMA-D/augmented107_log_mel_spectrograms_train'
val_data_path = '/workspace/dataset/CREMA-D/augmented107_log_mel_spectrograms_test'

# 데이터셋 및 데이터 로더 설정
train_dataset = CustomDataset(root_dir=train_data_path, transform=get_transform())
val_dataset = CustomDataset(root_dir=val_data_path, transform=get_transform())
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=2)

# 디바이스 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 그래프 그리기 및 저장 함수
def plot_combined_metrics(train_values, val_values, metric_name, title, save_path):
    plt.figure(figsize=(10, 5))
    plt.plot(train_values, label=f'Train {metric_name}')
    plt.plot(val_values, label=f'Validation {metric_name}')
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel(metric_name)
    plt.legend()
    plt.savefig(save_path)
    plt.close()

# 교사 및 학생 네트워크 초기화
teacher_net = Teacher(
    image_size=(128, 128),
    patch_size=(128, 1),
    num_classes=6,
    dim=256,
    depth=6,
    heads=5,
    mlp_dim=256,
    channels=3,  # RGB 채널 설정
    dropout=0.4,
    emb_dropout=0.4
).to(device)

student_net = Student(
    image_size=(128, 128),
    patch_size=(128, 1),
    num_classes=6,
    dim=256,
    depth=3,
    heads=5,
    mlp_dim=256,
    channels=3,  # RGB 채널 설정
    dropout=0.4,
    emb_dropout=0.4
).to(device)

# 손실 함수 및 옵티마이저 설정
criterion_ce = nn.CrossEntropyLoss().to(device)
criterion_l1 = nn.L1Loss().to(device)
optimizer_teacher = optim.Adam(teacher_net.parameters(), lr=1e-4, weight_decay=5e-4)
optimizer_student = optim.Adam(student_net.parameters(), lr=1e-4, weight_decay=5e-4)

# 학습률 스케줄러 설정
scheduler_teacher = optim.lr_scheduler.StepLR(optimizer_teacher, step_size=10, gamma=0.5)
scheduler_student = optim.lr_scheduler.StepLR(optimizer_student, step_size=10, gamma=0.5)

# Teacher Network 학습 함수
def train_teacher():
    epochs = 50
    best_loss = float('inf')
    epoch_losses = []
    epoch_accuracies = []
    val_losses = []
    val_accuracies = []

    for epoch in range(epochs):
        teacher_net.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            # Forward pass
            x = teacher_net.encoder(images)
            outputs, features = teacher_net.decoder(x)
            loss = criterion_ce(outputs, labels)
            
            # Backpropagation 및 최적화
            optimizer_teacher.zero_grad()
            loss.backward()
            optimizer_teacher.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        epoch_loss = running_loss / len(train_loader)
        epoch_accuracy = 100 * correct / total
        epoch_losses.append(epoch_loss)
        epoch_accuracies.append(epoch_accuracy)

        # Validation loss 및 accuracy 계산
        teacher_net.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for val_images, val_labels in val_loader:
                val_images, val_labels = val_images.to(device), val_labels.to(device)
                x = teacher_net.encoder(val_images)
                val_outputs, features = teacher_net.decoder(x)
                v_loss = criterion_ce(val_outputs, val_labels)
                val_loss += v_loss.item()
                _, val_predicted = torch.max(val_outputs, 1)
                val_total += val_labels.size(0)
                val_correct += (val_predicted == val_labels).sum().item()
        
        val_loss /= len(val_loader)
        val_accuracy = 100 * val_correct / val_total
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)

        print(f"Teacher Epoch [{epoch+1}/{epochs}] | Loss: {epoch_loss:.4f} | Accuracy: {epoch_accuracy:.2f}% | Validation Loss: {val_loss:.4f} | Validation Accuracy: {val_accuracy:.2f}%")

        # 성능 개선 시 모델 저장
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(teacher_net.state_dict(), os.path.join(TEACHER_DIR, 'teacher_model.pth'))
            print(f"Improved Teacher model saved at epoch {epoch+1}")
        
        # 학습률 스케줄러 업데이트
        scheduler_teacher.step()

    # 그래프 그리기 및 저장
    plot_combined_metrics(epoch_losses, val_losses, 'Loss', 'Teacher Network Loss', os.path.join(TEACHER_DIR, 'teacher_loss_plot.png'))
    plot_combined_metrics(epoch_accuracies, val_accuracies, 'Accuracy', 'Teacher Network Accuracy', os.path.join(TEACHER_DIR, 'teacher_accuracy_plot.png'))

# Student Network 학습 함수
def train_student():
    epochs = 50
    alpha = 10
    best_loss = float('inf')
    epoch_losses = []
    epoch_accuracies = []
    val_losses = []
    val_accuracies = []

    # 교사 네트워크의 가중치 로드 및 평가 모드 설정
    teacher_net.load_state_dict(torch.load(os.path.join(TEACHER_DIR, 'teacher_model.pth')))
    teacher_net.eval()

    for epoch in range(epochs):
        student_net.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            # Teacher Network의 feature 계산 (Gradient 계산하지 않음)
            with torch.no_grad():
                teacher_features = teacher_net.encoder(images)
            
            # Student Network의 forward pass
            student_features = student_net.encoder(images)
            outputs, _ = student_net.decoder(student_features)
            loss_ce = criterion_ce(outputs, labels)
            loss_l1 = criterion_l1(student_features, teacher_features)
            loss = loss_ce + alpha * loss_l1
            
            # Backpropagation 및 최적화
            optimizer_student.zero_grad()
            loss.backward()
            optimizer_student.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        epoch_loss = running_loss / len(train_loader)
        epoch_accuracy = 100 * correct / total
        epoch_losses.append(epoch_loss)
        epoch_accuracies.append(epoch_accuracy)

        # Validation loss 및 accuracy 계산
        student_net.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for val_images, val_labels in val_loader:
                val_images, val_labels = val_images.to(device), val_labels.to(device)
                student_features = student_net.encoder(val_images)
                val_outputs, _ = student_net.decoder(student_features)
                v_loss = criterion_ce(val_outputs, val_labels)
                val_loss += v_loss.item()
                _, val_predicted = torch.max(val_outputs, 1)
                val_total += val_labels.size(0)
                val_correct += (val_predicted == val_labels).sum().item()
        
        val_loss /= len(val_loader)
        val_accuracy = 100 * val_correct / val_total
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)

        print(f"Student Epoch [{epoch+1}/{epochs}] | Loss: {epoch_loss:.4f} | Accuracy: {epoch_accuracy:.2f}% | Validation Loss: {val_loss:.4f} | Validation Accuracy: {val_accuracy:.2f}%")

        # 성능 개선 시 모델 저장
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(student_net.state_dict(), os.path.join(STUDENT_DIR, 'student_model.pth'))
            print(f"Improved Student model saved at epoch {epoch+1}")
        
        # 학습률 스케줄러 업데이트
        scheduler_student.step()

    # 그래프 그리기 및 저장
    plot_combined_metrics(epoch_losses, val_losses, 'Loss', 'Student Network Loss', os.path.join(STUDENT_DIR, 'student_loss_plot.png'))
    plot_combined_metrics(epoch_accuracies, val_accuracies, 'Accuracy', 'Student Network Accuracy', os.path.join(STUDENT_DIR, 'student_accuracy_plot.png'))

# Teacher Network 학습
train_teacher()

# Student Network 학습
train_student()
