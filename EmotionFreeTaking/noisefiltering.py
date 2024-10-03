import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import torch
if torch.cuda.is_available():
        device = torch.device('cuda')

print(device)

import librosa
import soundfile as sf
import noisereduce as nr
import numpy as np

#1024 , 256 , 32, 1000 , 1 / 2048 , 512 , 128, 2000 , 1
def reduce_noise(input_file, output_file, n_fft=2048, hop_length = 512 , time_mask_smooth_ms=128 , freq_mask_smooth_hz=2000 , prop_decrease=1):
    audio_data, sr = librosa.load(input_file, sr=None)

    reduced_noise = nr.reduce_noise(y=audio_data, sr=sr, n_fft=n_fft , hop_length = hop_length , time_mask_smooth_ms= time_mask_smooth_ms , freq_mask_smooth_hz=freq_mask_smooth_hz , prop_decrease=prop_decrease)
    
    sf.write(output_file, reduced_noise, sr)
    print(f"노이즈 및 에코 제거된 파일 저장 완료: {output_file}")

def process_directory(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.endswith(".wav"):
                input_path = os.path.join(root, file)
                output_path = os.path.join(output_dir, file)
                reduce_noise(input_path, output_path)

input_directory = '/workspace/dataset/free_talking_datasets/voice_train'
output_directory = '/workspace/dataset/free_talking_datasets/filtered_voice_train'

process_directory(input_directory, output_directory)
