import matplotlib.pyplot as plt
from models import *
from Segdataset import read_test_file_list, SegDataset
import argparse
from utils import plot_confusion_matrix, get_evaluation
from torch.utils.data import DataLoader
import os
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--data_root', type=str, default=r'datasets/test', help='root of data')
parser.add_argument('--save_root', type=str, default=r'checkpoints/CNN', help='root of saved confusion_matrix')
parser.add_argument('--random_seed', type=int, default=10, help='random seed')
parser.add_argument('--model_kind', type=str, default='cnn', help='kind of model, i.e. cnn or rnn or transformer')
parser.add_argument('--num_workers', type=int, default=8, help='num_workers')
parser.add_argument('--n_mfcc', type=int, default=16, help='characteristic dimension of MFCC')
parser.add_argument('--n_folds', type=int, default=5, help='number of folds')

opt = parser.parse_args()

def main():
    emotions = ['NotStressed', 'Stressed']
    all_metrics = []

    for fold in range(opt.n_folds):
        if opt.model_kind == 'cnn':
            model = CNN_test()

        output_emotion_list = []
        label_emotion_list = []

        mfcc_list, emotion_list = read_test_file_list(root=opt.data_root, n_mfcc=opt.n_mfcc)
        test_set = SegDataset(mfcc_list, emotion_list)
        test_iter = DataLoader(test_set, batch_size=1, drop_last=True, num_workers=opt.num_workers)

        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')
    
        print(f"Using device: {device}")

        model = model.to(device)

        # 모델 로드
        model.load_state_dict(torch.load(os.path.join(opt.save_root, f'fold_{fold + 1}_best_epoch.pth'), map_location=device))
        model.eval()

        with torch.no_grad():
            for index, (mfcc, emotion) in enumerate(test_iter):
                mfcc, emotion = mfcc.to(device), emotion.to(device)
                output_emotion = model(mfcc)
                output_emotion = torch.sigmoid(output_emotion)
                predicted_emotion = (output_emotion > 0.56).float()

                output_emotion_list.append(predicted_emotion.cpu().numpy())
                label_emotion_list.append(emotion.cpu().numpy())

        output_emotion_list = np.vstack(output_emotion_list)[:, 0]
        label_emotion_list = np.vstack(label_emotion_list)[:, 0]

        # 평가
        acc, precision, recall, f1 = get_evaluation(label_emotion_list, output_emotion_list)
        all_metrics.append((acc, precision, recall, f1))

        # 결과 출력
        print(f"Fold {fold + 1} - Accuracy: {acc}, Precision: {precision}, Recall: {recall}, F1 Score: {f1}")

        # Confusion Matrix 생성 및 저장
        plt.figure()
        plot_confusion_matrix(label_emotion_list, output_emotion_list, emotions)
        plt.title(f'Confusion Matrix - Fold {fold + 1}')
        plt.savefig(os.path.join(opt.save_root, f'confusion_matrix_fold_{fold + 1}.png'))
        plt.show()

    # 평균값 계산
    avg_metrics = np.mean(all_metrics, axis=0)
    print(f"Average - Accuracy: {avg_metrics[0]}, Precision: {avg_metrics[1]}, Recall: {avg_metrics[2]}, F1 Score: {avg_metrics[3]}")

if __name__ == '__main__':
    main()
