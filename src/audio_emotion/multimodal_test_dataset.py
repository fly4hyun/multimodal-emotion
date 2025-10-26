import warnings
warnings.filterwarnings(action='ignore')

import pandas as pd
import os
from tqdm import tqdm
import random
import shutil
import zipfile

from sklearn.model_selection import train_test_split

import torch
import torchaudio

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

# 34406  가지 flagship
# 10267  가지 studio
# 9432   가지 multi_modal_data
# 54773  가지 multimodal_video

name____ = 'multimodal_video'

data_path_CP949 = [#'flagship', 
                   #'multi_modal_data', 
                   #'studio'
                   ]

data_path_UTF_8 = ['multimodal_video', 
                   ]

data_names = []
data_paths = []
labels = []
genders = []
texts = []

print()
print('***  data loading ....  ***')

for datasetname in data_path_CP949:
    ## labels.csv 파일을 dict 형식으로 저장
    print(datasetname)
    data_info = pd.read_csv(os.path.join(datasetname, "labels.csv"), encoding = "CP949").to_dict(orient = 'list')

    ## 데이터 이름과 해당 데이터의 라벨값 저장
    data_names_one = data_info["audio"]
    labels_one = data_info["label"]
    genders_one = data_info["gender"]
    tests_one = data_info["text"]

    ## 학습하기 위한 감정을 제외한 다른 라벨 제거
    data_names_temp = []
    data_paths_temp = []
    labels_temp = []
    genders_temp = []
    texts_temp = []
    for i in range(len(labels_one)):
        if labels_one[i] in label_7_mapping.keys():
            data_names_temp.append(data_names_one[i])
            data_paths_temp.append(os.path.join(datasetname, 'audio', data_names_one[i]))
            labels_temp.append(labels_one[i])
            genders_temp.append(genders_one[i])
            texts_temp.append(tests_one[i])

    data_names = data_names + data_names_temp
    data_paths = data_paths + data_paths_temp
    labels = labels + labels_temp
    genders = genders + genders_temp
    texts = texts + texts_temp
    
for datasetname in data_path_UTF_8:
    ## labels.csv 파일을 dict 형식으로 저장
    print(datasetname)
    data_info = pd.read_csv(os.path.join(datasetname, "labels.csv"), encoding = "UTF-8").to_dict(orient = 'list')

    ## 데이터 이름과 해당 데이터의 라벨값 저장
    data_names_one = data_info["audio"]
    labels_one = data_info["label"]
    genders_one = data_info["gender"]
    tests_one = data_info["text"]

    ## 학습하기 위한 감정을 제외한 다른 라벨 제거
    data_names_temp = []
    data_paths_temp = []
    labels_temp = []
    genders_temp = []
    texts_temp = []
    for i in range(len(labels_one)):
        if labels_one[i] in label_7_mapping.keys():
            data_names_temp.append(data_names_one[i])
            data_paths_temp.append(os.path.join(datasetname, 'audio', data_names_one[i]))
            labels_temp.append(labels_one[i])
            genders_temp.append(genders_one[i])
            texts_temp.append(tests_one[i])

    data_names = data_names + data_names_temp
    data_paths = data_paths + data_paths_temp
    labels = labels + labels_temp
    genders = genders + genders_temp
    texts = texts + texts_temp

print('***  data loading Done  ***')
print()

print('***  data selecting (1) ....  ***')

label_7_conut = {
    "ang": 0,   # 분노
    "dis": 0,   # 싫음
    "fea": 0,   # 두려움
    "hap": 0,   # 행복
    "sad": 0,   # 슬픔
    "sur": 0,   # 놀람
    "neu": 0,   # 중립
}

labels_order = ["ang", "dis", "fea", "hap", "sad", "sur", "neu"]
emotion_valid_data_name = [[], [], [], [], [], [], []]
emotion_valid_data_path = [[], [], [], [], [], [], []]
emotion_valid_data_label = [[], [], [], [], [], [], []]
emotion_valid_data_gender = [[], [], [], [], [], [], []]
emotion_valid_data_text = [[], [], [], [], [], [], []]

emotion_train_data_name = [[], [], [], [], [], [], []]
emotion_train_data_path = [[], [], [], [], [], [], []]
emotion_train_data_label = [[], [], [], [], [], [], []]
emotion_train_data_gender = [[], [], [], [], [], [], []]
emotion_train_data_text = [[], [], [], [], [], [], []]

iterations = tqdm(range(len(data_paths)))

for i in iterations:
    
    if labels[i] in list(label_7_mapping.keys()):
        
        wave_form, sample_rate = torchaudio.load(data_paths[i], normalize=True)
        audio_len = wave_form.size(1)
        
        if audio_len >= sample_rate and audio_len < sample_rate * 10:
            
            label_7_conut[labels[i]] += 1
            emotion_valid_data_name[label_7_mapping[labels[i]]].append(data_names[i])
            emotion_valid_data_path[label_7_mapping[labels[i]]].append(data_paths[i])
            emotion_valid_data_label[label_7_mapping[labels[i]]].append(labels[i])
            emotion_valid_data_gender[label_7_mapping[labels[i]]].append(genders[i])
            emotion_valid_data_text[label_7_mapping[labels[i]]].append(texts[i])
            
        else:
            emotion_train_data_name[label_7_mapping[labels[i]]].append(data_names[i])
            emotion_train_data_path[label_7_mapping[labels[i]]].append(data_paths[i])
            emotion_train_data_label[label_7_mapping[labels[i]]].append(labels[i])
            emotion_train_data_gender[label_7_mapping[labels[i]]].append(genders[i])
            emotion_train_data_text[label_7_mapping[labels[i]]].append(texts[i])
            

print(label_7_conut)

print('***  data selecting (1) Done  ***')
print()

print('***  data selecting (2) ....  ***')

data_dcit = {}
min_value = int(min(label_7_conut.values()) / 10)

for i in range(7):
    nums = len(emotion_valid_data_label[i])
    index_list = [i for i in range(nums)]
    rand_index = random.sample(index_list, min_value)
    
    iterations = tqdm(range(nums))
    for j in iterations:
        
        if j in rand_index:

            data_one_info = [emotion_valid_data_label[i][j], emotion_valid_data_gender[i][j], emotion_valid_data_text[i][j], "test"]

            data_dcit[emotion_valid_data_name[i][j]] = data_one_info
            
        else:
            
            emotion_train_data_name[label_7_mapping[labels[i]]].append(emotion_valid_data_name[i][j])
            emotion_train_data_path[label_7_mapping[labels[i]]].append(emotion_valid_data_path[i][j])
            emotion_train_data_label[label_7_mapping[labels[i]]].append(emotion_valid_data_label[i][j])
            emotion_train_data_gender[label_7_mapping[labels[i]]].append(emotion_valid_data_gender[i][j])
            emotion_train_data_text[label_7_mapping[labels[i]]].append(emotion_valid_data_text[i][j])
            
        
        #shutil.copyfile(emotion_valid_data_path[i][nums_14000], './TrVTe_dataset/audio/' + emotion_valid_data_name[i][nums_14000])


data_train_dcit = {}

for i in range(7):
    nums = len(emotion_train_data_label[i])
    
    iterations = tqdm(range(nums))
    for j in iterations:

        data_one_info = [emotion_train_data_label[i][j], emotion_train_data_gender[i][j], emotion_train_data_text[i][j], "train"]

        data_train_dcit[emotion_train_data_name[i][j]] = data_one_info



print('***  data selecting (2) Done  ***')
print()

print('***  data saving ....  ***')

column = ['label', 'gender', 'text', 'TrVTe']

df = pd.DataFrame.from_dict(data = data_dcit, orient = 'index', columns = column)
df.index.name = 'audio'
df.to_csv('./multimodal_test/test_' + name____ + '.csv', encoding = 'UTF-8')

df_train = pd.DataFrame.from_dict(data = data_train_dcit, orient = 'index', columns = column)
df_train.index.name = 'audio'
df_train.to_csv('./multimodal_test/train_' + name____ + '.csv', encoding = 'UTF-8')

print('***  data saving Done  ***')
print()






