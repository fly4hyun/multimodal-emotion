import warnings
warnings.filterwarnings(action='ignore')

import pandas as pd
import os
from tqdm import tqdm

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


data_path_UTF_8 = ['dataset_for_emotional_speech', 
                   #'dialogue_speech_datasets_for_emotional_classification', 
                   #'emotion_and_speaker_style', 
                   #'emotion_syn_voice', 
                   #'flagship', 
                   #'multi_modal_data', 
                   #'multimodal_video', 
                   #'novel_audio_emotion', 
                   #'studio'
                   ]

data_names = []
labels = []

#UTF-8
#CP949


for datasetname in data_path_UTF_8:
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


#################################################

min_len = 100000000
max_len = 0
total_len = 0
total_num = 0

label_7_conut = {
    "ang": 0,   # 분노
    "dis": 0,   # 싫음
    "fea": 0,   # 두려움
    "hap": 0,   # 행복
    "sad": 0,   # 슬픔
    "sur": 0,   # 놀람
    "neu": 0,   # 중립
}

###############################################


for datasetname in data_path_UTF_8:
    
    iterations = tqdm(range(len(data_names_one)))
    for num_i in iterations:
        i = data_names_one[num_i]
        j = labels_one[num_i]
        

        
        if j in list(label_7_conut.keys()):
            label_7_conut[j] += 1
            
            wave_form, sample_rate = torchaudio.load(os.path.join(datasetname, 'audio', i), normalize=True)
            
            if wave_form.size(1) < min_len:
                min_len = wave_form.size(1)
            if wave_form.size(1) > max_len:
                max_len = wave_form.size(1)
            total_len = total_len + wave_form.size(1)
            total_num = total_num + 1
        
print('sample_rate', sample_rate)
print('max', max_len)
print('min', min_len)
print('avg', total_len / total_num)
print('num', total_num)

print(label_7_conut)


# ## csv 내 파일이 존재하는지 여부 확인 코드
# for datasetname in data_path_UTF_8:
#     er = 0
#     cor = 0
    
#     for i in data_names_one:
#         if os.path.exists(os.path.join(datasetname, 'audio', i)):
#             cor = cor + 1
#         else:
#             er = er + 1
#             print(i)
            
            
            
#     print(er)
#     print(cor)



#     aaa = os.listdir(datasetname + '/audio')

#     print(len(aaa))