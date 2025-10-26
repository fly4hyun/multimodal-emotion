import warnings
warnings.filterwarnings(action='ignore')

import pandas as pd
import os

from sklearn.model_selection import train_test_split

import torch
import torchaudio
from torch.utils.data import Dataset




## 7종 감정
label_7_mapping = {
    "ang": 0,   # 분노
    "dis": 1,   # 싫음
    "fea": 2,   # 두려움
    "hap": 3,   # 행복
    "sad": 4,   # 슬픔
    "sur": 5,   # 놀람
    "neu": 6,   # 중립
}

### 데이터셋 info

# 151680 가지 dataset_for_emotional_speech
# 42741  가지 dialogue_speech_datasets_for_emotional_classification
# 868838 가지 emotion_and_speaker_style
# 22087  가지 emotion_syn_voice
# 34406  가지 flagship
# 9432   가지 multi_modal_data
# 54773  가지 multimodal_video
# 259411 가지 novel_audio_emotion
# 10267  가지 studio
# 
# 총 1,453,635 가지
# max length : 4404144

## 데이터셋 폴더 위치
# data_path = "flagship"
data_path = ['dataset_for_emotional_speech', 
             'dialogue_speech_datasets_for_emotional_classification', 
             'emotion_and_speaker_style', 
             'emotion_syn_voice', 
             'flagship', 
             'multi_modal_data', 
             'multimodal_video', 
             'novel_audio_emotion', 
             'studio'
             ]

data_path_CP949 = ['dataset_for_emotional_speech', 
                   'dialogue_speech_datasets_for_emotional_classification', 
                   #'emotion_and_speaker_style', 
                   #'emotion_syn_voice', 
                   'flagship', 
                   'multi_modal_data', 
                   #'multimodal_video', 
                   'novel_audio_emotion', 
                   'studio'
                   ]

data_path_UTF_8 = [#'dataset_for_emotional_speech', 
                   #'dialogue_speech_datasets_for_emotional_classification', 
                   'emotion_and_speaker_style', 
                   'emotion_syn_voice', 
                   #'flagship', 
                   #'multi_modal_data', 
                   'multimodal_video', 
                   #'novel_audio_emotion', 
                   #'studio'
                   ]


data_names = []
labels = []

for datasetname in data_path_CP949:
    ## labels.csv 파일을 dict 형식으로 저장
    
    data_info = pd.read_csv(os.path.join(datasetname, "labels.csv"), encoding = "CP949").to_dict(orient = 'list')

    ## 데이터 이름과 해당 데이터의 라벨값 저장
    data_names_one = data_info["audio"]
    labels_one = data_info["label"]

    ## 학습하기 위한 감정을 제외한 다른 라벨 제거
    data_names_temp = []
    labels_temp = []
    for i in range(len(labels_one)):
        if labels_one[i] in label_7_mapping.keys():
            data_names_temp.append(os.path.join(datasetname, 'audio', data_names_one[i]))
            labels_temp.append(label_7_mapping[labels_one[i]])

    data_names = data_names + data_names_temp
    labels = labels + labels_temp
    
for datasetname in data_path_UTF_8:
    ## labels.csv 파일을 dict 형식으로 저장
    
    data_info = pd.read_csv(os.path.join(datasetname, "labels.csv"), encoding = "UTF-8").to_dict(orient = 'list')

    ## 데이터 이름과 해당 데이터의 라벨값 저장
    data_names_one = data_info["audio"]
    labels_one = data_info["label"]

    ## 학습하기 위한 감정을 제외한 다른 라벨 제거
    data_names_temp = []
    labels_temp = []
    for i in range(len(labels_one)):
        if labels_one[i] in label_7_mapping.keys():
            data_names_temp.append(os.path.join(datasetname, 'audio', data_names_one[i]))
            labels_temp.append(label_7_mapping[labels_one[i]])

    data_names = data_names + data_names_temp
    labels = labels + labels_temp
    


# ## train : validation : test = 8 : 1 : 1
# train_val_data_names, test_data_names, train_val_labels, test_labels = train_test_split(data_names, labels, test_size = 1/10, random_state = 444)
# train_data_names, val_data_names, train_labels, val_labels = train_test_split(train_val_data_names, train_val_labels, test_size = 1/9, random_state = 444)


## 데이터 10분의 1로 나누기
_, data_names_, _, labels_ = train_test_split(data_names, labels, test_size = 1/10, random_state = 444)
## train : validation : test = 8 : 1 : 1
train_val_data_names, test_data_names, train_val_labels, test_labels = train_test_split(data_names_, labels_, test_size = 1/10, random_state = 444)
train_data_names, val_data_names, train_labels, val_labels = train_test_split(train_val_data_names, train_val_labels, test_size = 1/9, random_state = 444)




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

    def __len__(self):
        
        return len(self.labels)
        
    def __getitem__(self, idx):
        
        wave_form, sample_rate = torchaudio.load(self.data_names[idx], normalize=True)
        
        #wave_form = torch.FloatTensor(wave_form[::10])
        wave_form = torch.nn.functional.pad(wave_form, (0, 199999 - wave_form.size(-1)), 'constant', 0.0)
        wave_form = wave_form[0]
        
        
        label = torch.tensor(self.labels[idx])
        #label = torch.nn.functional.one_hot(label, num_classes = 7)
        
        sample = {
            'audio': wave_form, 
            'sample_rate': sample_rate, 
            'label': label
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
    
    from torch.utils.data import DataLoader
    train_loader = DataLoader(trainset, batch_size = 1, shuffle = True, num_workers = 3, pin_memory = True)
    valid_loader = DataLoader(validset, batch_size = 1, shuffle = False, num_workers = 3, pin_memory = True)
    test_loader = DataLoader(testset, batch_size = 1, shuffle = False, num_workers = 3, pin_memory = True)

    max_len = 0
    for train_data in train_loader:
        
        train_audio = train_data['audio']
        
        
        if max_len < train_audio.size(2):
            max_len = train_audio.size(2)
            print(max_len)

    
    max_len_ = 0
    for train_data in valid_loader:
        
        train_audio = train_data['audio']
        
        
        if max_len_ < train_audio.size(2):
            max_len_ = train_audio.size(2)
            print(max_len_)
            
    max_len__ = 0
    for train_data in test_loader:
        
        train_audio = train_data['audio']
        
        
        if max_len__ < train_audio.size(2):
            max_len__ = train_audio.size(2)
            print(max_len__)
    
    
    print(max_len)
    print(max_len_)
    print(max_len__)
    
    

    train_data = next(iter(trainset))
        
    audio = train_data['audio']
    sample_rate = train_data['sample_rate']
    label = train_data['label']
        
    print('audio size: (', audio.size(0), 'x', audio.size(1), ')')
    print('sample rate: ', sample_rate)
    #print('emotion: ', label.item(), ' -> ', label_7_mapping_[int(label)])