import os
import shutil
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from PIL import Image

# 음성 길이 정규화 함수 (4초로 제한)
def normalize_voice_len(y, normalized_len=64000):  # 4초 * 16000Hz = 64000 샘플
    nframes = len(y)
    if nframes < normalized_len:
        res = normalized_len - nframes
        res_data = np.zeros(res, dtype=np.float32)
        y = np.concatenate((y, res_data), axis=0)
    else:
        y = y[:normalized_len]
    return y

# 노이즈 추가 함수
def add_noise(y, noise_factor=0.007):
    noise = np.random.randn(len(y))
    return y + noise_factor * noise

# 시간 이동 함수
def time_shift(y, shift_max=0.2, sr=16000, min_shift=0.05):
    shift = np.random.randint(sr * min_shift, sr * shift_max)
    direction = np.random.choice([-1, 1])
    shift = shift * direction
    # 음성이 실제로 이동하도록 보장
    if shift == 0:
        shift = sr * min_shift * direction
    return np.roll(y, shift)


# 속도 변화 함수 (speed perturbation)
def speed_perturbation(y):
    speed_factor = np.random.choice([0.8, 0.9, 1.1, 1.2])
    return librosa.effects.time_stretch(y.astype(np.float32), rate=speed_factor)

# 로그 멜 스펙트로그램 생성 및 저장 함수
def save_log_mel_spectrogram(file_path, output_dir, sr=16000, n_mels=128, img_size=(128, 128), augmentation_type="original"):
    # 음성 파일 로드
    y, _ = librosa.load(file_path, sr=sr)
    
    # 음성 길이 정규화
    y = normalize_voice_len(y)
    
    # 데이터 증강 수행
    if augmentation_type == "noise":
        y = add_noise(y)
    elif augmentation_type == "time_shift":
        y = time_shift(y)
    elif augmentation_type == "speed":
        y = speed_perturbation(y)
    
    # STFT를 통한 멜 스펙트로그램 생성
    mel_spect = librosa.feature.melspectrogram(
        y=y, 
        sr=sr, 
        n_mels=n_mels,
        n_fft=1024,          # 논문 설정
        hop_length=64,       # 논문 설정
        win_length=512,      # 논문 설정
        window='hamming'     # 논문 설정
    )
    log_mel_spect = librosa.power_to_db(mel_spect, ref=np.max)
    
    # 이미지 파일 이름 설정
    file_name = os.path.splitext(os.path.basename(file_path))[0]
    output_path = os.path.join(output_dir, f"{file_name}_{augmentation_type}.png")
    
    # 로그 멜 스펙트로그램 이미지 저장 (크기 조정 포함)
    plt.figure(figsize=(img_size[0] / 100, img_size[1] / 100), dpi=100)
    librosa.display.specshow(
        log_mel_spect, 
        sr=sr, 
        hop_length=64,  # 논문 설정
        x_axis='time', 
        y_axis='mel',
        cmap='magma'  # 컬러 맵을 'magma'로 변경하여 RGB 이미지 생성
    )
    plt.axis('off')
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close()
    
    # PIL을 사용하여 정확한 크기로 다시 리사이즈
    with Image.open(output_path) as img:
        img = img.resize(img_size, Image.LANCZOS)
        img.save(output_path)

# 전체 음성 파일 처리 함수
def process_all_audio_files(input_dir, output_dir):
    # output_dir이 존재하면 내부 파일 삭제
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)  # 폴더 삭제
    os.makedirs(output_dir)  # 새 폴더 생성
    
    for file_name in os.listdir(input_dir):
        if file_name.endswith('.wav'):
            file_path = os.path.join(input_dir, file_name)
            
            # 원본 저장
            save_log_mel_spectrogram(file_path, output_dir, augmentation_type="original")
            
            # 노이즈 추가 버전 저장
            save_log_mel_spectrogram(file_path, output_dir, augmentation_type="noise")
            
            # 시간 이동 버전 저장
            save_log_mel_spectrogram(file_path, output_dir, augmentation_type="time_shift")
            
            # 속도 변화 버전 저장
            save_log_mel_spectrogram(file_path, output_dir, augmentation_type="speed")

# 입력 디렉토리와 출력 디렉토리 설정
input_directories = {
    'train': '/workspace/dataset/CREMA-D/train',
    'test': '/workspace/dataset/CREMA-D/test'
}

output_directories = {
    'train': '/workspace/dataset/CREMA-D/augmented107_log_mel_spectrograms_train',
    'test': '/workspace/dataset/CREMA-D/augmented107_log_mel_spectrograms_test'
}

# 각 데이터셋에 대해 처리
for dataset_type in ['train', 'test']:
    input_dir = input_directories[dataset_type]
    output_dir = output_directories[dataset_type]
    process_all_audio_files(input_dir, output_dir)
