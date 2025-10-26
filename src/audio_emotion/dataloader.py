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

train_data_names = []
val_data_names = []
test_data_names = []

train_labels = []
val_labels = []
test_labels = []

### flagship ###

data_name_list = ['flagship', 'studio', 'multi_modal_data', 'multimodal_video']

for dataset_name in data_name_list:
    data_names = []
    labels = []

    data_info = pd.read_csv("./emotion_data_csv/train_" + dataset_name + ".csv", encoding = "UTF-8").to_dict(orient = 'list')
    data_names_one = data_info["audio"]
    labels_one = data_info["label"]
    trvtes_one = data_info["TrVTe"]

    for i in range(len(trvtes_one)):

        data_names.append("./" + dataset_name + "/audio/" + data_names_one[i])
        labels.append(label_7_mapping[labels_one[i]])

    tr_data_names, va_data_names, tr_labels, va_labels = train_test_split(data_names, labels, test_size = 1/10, random_state = 444)
    train_data_names = train_data_names + tr_data_names
    val_data_names = val_data_names + va_data_names
    train_labels = train_labels + tr_labels
    val_labels = val_labels + va_labels

    data_names = []
    labels = []

    data_info = pd.read_csv("./emotion_data_csv/test_" + dataset_name + ".csv", encoding = "UTF-8").to_dict(orient = 'list')
    data_names_one = data_info["audio"]
    labels_one = data_info["label"]
    trvtes_one = data_info["TrVTe"]

    for i in range(len(trvtes_one)):

        data_names.append("./" + dataset_name + "/audio/" + data_names_one[i])
        labels.append(label_7_mapping[labels_one[i]])

    test_data_names = test_data_names + data_names
    test_labels = test_labels + labels








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
            
        self.resampler_48000 = T.Resample(48000, 16000)
        self.resampler_44100 = T.Resample(44100, 16000)
            
        MODEL_ID = "jonatasgrosman/wav2vec2-large-xlsr-53-english"
        
        self.processor = Wav2Vec2Processor.from_pretrained(MODEL_ID)
        
        self.processor_astm = AutoProcessor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
        
    def __len__(self):
        
        return len(self.labels)
        
    def __getitem__(self, idx):
        
        wave_form, sample_rate = torchaudio.load(self.data_names[idx], normalize=True)
        
        if sample_rate == 44100:
            wave_form = self.resampler_44100(wave_form)
            sample_rate = 16000
        if sample_rate == 48000:
            wave_form = self.resampler_44100(wave_form)
            sample_rate = 16000
            
        wave_form_ = self.processor(wave_form, sampling_rate=sample_rate, return_tensors="pt", padding=True)
        input_values = wave_form_.input_values
        attention_mask = wave_form_.attention_mask
            
        input_values = input_values.squeeze(0)
        attention_mask = attention_mask.squeeze(0)
        attention_mask = torch.nn.functional.pad(attention_mask, (0, input_values.size(-1)), 'constant', 1.0)
        
        wave_form = torch.nn.functional.pad(wave_form, (0, 160000 - wave_form.size(-1)), 'constant', 0.0)
        input_values = torch.nn.functional.pad(input_values, (0, 160000 - input_values.size(-1)), 'constant', 0.0)
        attention_mask = torch.nn.functional.pad(attention_mask, (0, 160000 - attention_mask.size(-1)), 'constant', 0.0)

        label = torch.tensor(self.labels[idx])
        #label = torch.nn.functional.one_hot(label, num_classes = 7)
        
        sample = {
            'audio': wave_form[0], 
            'sample_rate': sample_rate, 
            'label': label, 
            'input_values':input_values[0], 
            'attention_mask':attention_mask, 
        }
        
        return sample
    
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