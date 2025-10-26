import warnings
warnings.filterwarnings(action='ignore')

import pandas as pd
import os

from sklearn.model_selection import train_test_split

import torch
import torchaudio
from torch.utils.data import Dataset
import torchaudio.transforms as T

from transformers import Wav2Vec2Processor, AutoProcessor

## 7종 감정
label_7_mapping = {
    "ang": 0,   # 분노, angry
    "dis": 1,   # 싫음, disgust
    "fea": 2,   # 두려움, fear
    "hap": 3,   # 행복, happy
    "sad": 4,   # 슬픔, sad
    "sur": 5,   # 놀람, surprise
    "neu": 6,   # 중립, neutral
}

data_path = 'TrVTe_dataset'

data_names = []
labels = []

data_info = pd.read_csv(os.path.join(data_path, "labels.csv"), encoding = "UTF-8").to_dict(orient = 'list')

## 데이터 이름과 해당 데이터의 라벨값 저장
data_names_one = data_info["audio"]
labels_one = data_info["label"]
trvtes_one = data_info["TrVTe"]

train_data_names = []
val_data_names = []
test_data_names = []

train_labels = []
val_labels = []
test_labels = []

for i in range(len(trvtes_one)):
    if trvtes_one[i] == "train":
        train_data_names.append(os.path.join(data_path, 'audio', data_names_one[i]))
        train_labels.append(label_7_mapping[labels_one[i]])
    elif trvtes_one[i] == "valid":
        val_data_names.append(os.path.join(data_path, 'audio', data_names_one[i]))
        val_labels.append(label_7_mapping[labels_one[i]])
    elif trvtes_one[i] == "test":
        test_data_names.append(os.path.join(data_path, 'audio', data_names_one[i]))
        test_labels.append(label_7_mapping[labels_one[i]])
        



class CustomDataset(Dataset):
    def __init__(self, data_type = 'train'):
        
        if data_type == 'train':
            self.data_names = train_data_names
            self.labels = train_labels
        elif data_type == 'valid':
            self.data_names = val_data_names
            self.labels = val_labels
        elif data_type == 'test':
            self.data_names = test_data_names
            self.labels = test_labels
            
        self.resampler_48000 = T.Resample(48000, 48000)
        self.resampler_44100 = T.Resample(44100, 48000)
        
        ## librosa mfcc와 동일하게 만들기 위해
        window_size = 1024
        hop_length = 512
        n_mels = 128
        self.mfcc_48000 = torchaudio.transforms.MFCC(sample_rate = 48000, 
                                                     n_mfcc=39, 
                                                     melkwargs={'n_fft': window_size, 'n_mels': n_mels, 'hop_length': hop_length})
        self.mfcc_44100 = torchaudio.transforms.MFCC(sample_rate = 44100, 
                                                     n_mfcc=39, 
                                                     melkwargs={'n_fft': window_size, 'n_mels': n_mels, 'hop_length': hop_length})

    def __len__(self):
        
        return len(self.labels)
        
    def __getitem__(self, idx):
        
        wave_form, sample_rate = torchaudio.load(self.data_names[idx], normalize=True)
        
        if sample_rate == 44100:
            wave_form = self.resampler_44100(wave_form)
            sample_rate = 48000
        wave_form = torch.nn.functional.pad(wave_form, (0, 480000 - wave_form.size(-1)), 'constant', 0.0)
        # if sample_rate == 44100:
        #     wave_form = self.mfcc_44100(wave_form[0])
        # else:
        #     wave_form = self.mfcc_48000(wave_form[0])
        # wave_form = self.normalize_mfcc(wave_form)
        # #wave_form = wave_form.transpose(0, 1)
        
        label = torch.tensor(self.labels[idx])
        #label = torch.nn.functional.one_hot(label, num_classes = 7)
        
        sample = {
            'audio': wave_form[0], 
            'sample_rate': sample_rate, 
            'label': label, 
            # 'size_1': wave_form.size(0), 
            # 'size_2': wave_form.size(1),
        }
        
        return sample
    
    def normalize_mfcc(self, mfcc):
        mean = torch.mean(mfcc, dim=1, keepdim=True)
        std = torch.std(mfcc, dim=1, keepdim=True)
        normalized_mfcc = (mfcc - mean) / (std + 1e-8)  # Adding a small value to prevent division by zero
        return normalized_mfcc

trainset = CustomDataset('train')
validset = CustomDataset('valid')
testset = CustomDataset('test')


## 데이터 확인
if __name__ == '__main__':
    
    label_7_mapping_ = {
        0: "ang",   # 분노
        1: "dis",   # 싫음
        2: "fea",   # 두려움
        3: "hap",   # 행복
        4: "sad",   # 슬픔
        5: "sur",   # 놀람
        6: "neu",   # 중립
    }

    train_data = next(iter(trainset))
        
    audio = train_data['audio']
    sample_rate = train_data['sample_rate']
    label = train_data['label']
        
    print('audio size: (', audio.size(0), 'x', audio.size(1), ')')
    print('sample rate: ', sample_rate)
    print('emotion: ', label.item(), ' -> ', label_7_mapping_[int(label)])