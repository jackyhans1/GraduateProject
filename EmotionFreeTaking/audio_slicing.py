# import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# import json
# import torch
# import torchaudio
# from tqdm import tqdm  # tqdm 라이브러리 추가

# # GPU 설정
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print(f"Using device: {device}")

# train_data_path = '/workspace/dataset/free_talking_datasets/voice_test'
# label_data_path = '/workspace/dataset/free_talking_datasets/label_test'

# emotion_directories = {
#     "놀라움": "surprise",
#     "기쁨": "happy",
#     "두려움": "fear",
#     "사랑스러움": "lovely",
#     "슬픔": "sad",
#     "화남": "angry",
#     "없음": "neutral"
# }

# # 감정별 디렉토리 생성
# for emotion in emotion_directories.values():
#     os.makedirs(f'/workspace/dataset/free_talking_datasets/split_voice_test/{emotion}', exist_ok=True)

# def process_audio_file(audio_file, label_file):
#     # 오디오 파일 로드
#     waveform, sample_rate = torchaudio.load(audio_file)
#     waveform = waveform.to(device)

#     # 레이블 파일 로드
#     with open(label_file, 'r', encoding='utf-8') as f:
#         label_data = json.load(f)

#     for segment in label_data["Conversation"]:
#         # 쉼표를 제거하고 숫자로 변환
#         start_time = float(segment["StartTime"].replace(',', '')) * sample_rate
#         end_time = float(segment["EndTime"].replace(',', '')) * sample_rate
#         emotion = segment["SpeakerEmotionTarget"]

#         # 5초 미만의 오디오 구간은 제외
#         if (end_time - start_time) < 5 * sample_rate:
#             continue

#         # 감정 레이블이 매칭되지 않는 경우 제외
#         if emotion not in emotion_directories:
#             continue

#         # 오디오 구간 추출
#         segment_waveform = waveform[:, int(start_time):int(end_time)]

#         # 저장 경로 설정
#         emotion_dir = f'/workspace/dataset/free_talking_datasets/split_voice_test/{emotion_directories[emotion]}'
#         output_file = os.path.join(
#             emotion_dir,
#             f'{os.path.basename(audio_file).replace(".wav", "")}_{segment["TextNo"]}_{emotion_directories[emotion]}.wav'
#         )

#         # 오디오 파일 저장
#         torchaudio.save(output_file, segment_waveform.cpu(), sample_rate)
#         print(f"Saved: {output_file}")

# # 오디오 파일 처리
# wav_files = [f for f in os.listdir(train_data_path) if f.endswith(".wav")]
# for wav_file in tqdm(wav_files, desc="Processing audio files"):
#     audio_file_path = os.path.join(train_data_path, wav_file)
#     label_file_path = os.path.join(label_data_path, wav_file.replace(".wav", ".json"))

#     if os.path.exists(label_file_path):
#         process_audio_file(audio_file_path, label_file_path)
