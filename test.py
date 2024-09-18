import matplotlib.pyplot as plt
from models import *
from Segdataset import SegDataset
import argparse
from utils import plot_confusion_matrix, get_evaluation
from torch.utils.data import DataLoader
import os
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--data_root', type=str, default=r'datasets', help='root of data')
parser.add_argument('--log_root', type=str, default=r'checkpoints/CNN/ep049-loss0.449-val_loss0.451.pth', help='root of model.pth')
parser.add_argument('--save_root', type=str, default=r'checkpoints/CNN', help='root of saved confusion_matrix')
parser.add_argument('--random_seed', type=int, default=10, help='random seed')
parser.add_argument('--model_kind', type=str, default='cnn', help='kind of model, i.e. cnn or rnn or transformer')
parser.add_argument('--test_split', type=float, default=0.25, help='ratio of the test set')
parser.add_argument('--num_workers', type=int, default=8, help='num_workers')
parser.add_argument('--n_mfcc', type=int, default=16, help='characteristic dimension of MFCC')
parser.add_argument('--is_plt', type=bool, default=False, help='plt or not')
opt = parser.parse_args()

def main():
    emotions = ['NotStressed', 'Stressed']

    if opt.model_kind == 'cnn':
        model = CNN_test()
        
    output_emotion_list = []
    label_emotion_list = []
    
    if not os.path.exists(opt.save_root):
        os.makedirs(opt.save_root)
        print('create' + opt.save_root)

    val = SegDataset(root=opt.data_root, type='val', n_mfcc=opt.n_mfcc, random_state=opt.random_seed, test_size=opt.test_split)

    test_iter = torch.utils.data.DataLoader(val, batch_size=1, drop_last=True, num_workers=opt.num_workers)

    if torch.cuda.is_available():
        device = torch.device('cuda:1')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    model = model.to(device)
    model.load_state_dict(torch.load(opt.log_root, map_location=device))
    model.eval()

    with torch.no_grad():
        for index, (mfcc, emotion) in enumerate(test_iter):
            mfcc, emotion = mfcc.to(device), emotion.to(device)

            output_emotion = model(mfcc)
            output_emotion = output_emotion.float()  # mps -> 시그모이드 함수 int64 지원 안 함
            output_emotion = torch.sigmoid(output_emotion)
            
            # 0.5 기준으로 이진 분류
            predicted_emotion = (output_emotion > 0.56).float()

            # 리스트에 저장
            output_emotion_list.append(predicted_emotion.cpu().numpy())
            label_emotion_list.append(emotion.cpu().numpy())

    # 리스트를 numpy 배열로 변환 및 차원 축소
    output_emotion_list = np.vstack(output_emotion_list)  # 2차원 배열을 만들기 위해 vstack 사용
    label_emotion_list = np.vstack(label_emotion_list)

    # 차원 맞추기 (output_emotion_list의 두 번째 열 선택하여 1차원으로)
    output_emotion_list = output_emotion_list[:, 0]
    label_emotion_list = label_emotion_list[:, 0]

    # 평가
    acc, precision, recall, f1 = get_evaluation(label_emotion_list, output_emotion_list)
    with open(os.path.join(opt.save_root, 'evaluation.txt'), 'w') as f:
        f.write('accuracy: ' + str(acc))
        f.write('\r\n')
        f.write('precision: ' + str(precision))
        f.write('\r\n')
        f.write('recall: ' + str(recall))
        f.write('\r\n')
        f.write('f1 score: ' + str(f1))

    # Confusion Matrix 생성 및 저장
    plot_confusion_matrix(label_emotion_list, output_emotion_list, emotions)
    plt.savefig(opt.save_root + '/confusion_matrix_emotion.png')



if __name__ == '__main__':
    main()
