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
from torch.utils.data import Dataset




# ## 7종 감정
# label_7_mapping = {
#     "ang": 0,   # 분노
#     "dis": 1,   # 싫음
#     "fea": 2,   # 두려움
#     "hap": 3,   # 행복
#     "sad": 4,   # 슬픔
#     "sur": 5,   # 놀람
#     "neu": 6,   # 중립
# }

# ### 데이터셋 info

# # 151680 가지 dataset_for_emotional_speech
# # 42741  가지 dialogue_speech_datasets_for_emotional_classification
# # 868838 가지 emotion_and_speaker_style
# # 22087  가지 emotion_syn_voice
# # 34406  가지 flagship
# # 9432   가지 multi_modal_data
# # 54773  가지 multimodal_video
# # 259411 가지 novel_audio_emotion
# # 10267  가지 studio
# # 
# # 총 1,453,635 가지
# # max length : 4404144

# ## 데이터셋 폴더 위치
# # data_path = "flagship"
# data_path = ['dataset_for_emotional_speech', 
#              'dialogue_speech_datasets_for_emotional_classification', 
#              'emotion_and_speaker_style', 
#              'emotion_syn_voice', 
#              'flagship', 
#              'multi_modal_data', 
#              'multimodal_video', 
#              'novel_audio_emotion', 
#              'studio'
#              ]

# data_path_CP949 = ['dataset_for_emotional_speech', 
#                    'dialogue_speech_datasets_for_emotional_classification', 
#                    #'emotion_and_speaker_style', 
#                    #'emotion_syn_voice', 
#                    'flagship', 
#                    'multi_modal_data', 
#                    #'multimodal_video', 
#                    'novel_audio_emotion', 
#                    'studio'
#                    ]

# data_path_UTF_8 = [#'dataset_for_emotional_speech', 
#                    #'dialogue_speech_datasets_for_emotional_classification', 
#                    'emotion_and_speaker_style', 
#                    'emotion_syn_voice', 
#                    #'flagship', 
#                    #'multi_modal_data', 
#                    'multimodal_video', 
#                    #'novel_audio_emotion', 
#                    #'studio'
#                    ]


# data_names = []
# data_paths = []
# labels = []
# genders = []
# texts = []

# print()
# print('***  data loading ....  ***')

# for datasetname in data_path_CP949:
#     ## labels.csv 파일을 dict 형식으로 저장
#     print(datasetname)
#     data_info = pd.read_csv(os.path.join(datasetname, "labels.csv"), encoding = "CP949").to_dict(orient = 'list')

#     ## 데이터 이름과 해당 데이터의 라벨값 저장
#     data_names_one = data_info["audio"]
#     labels_one = data_info["label"]
#     genders_one = data_info["gender"]
#     tests_one = data_info["text"]

#     ## 학습하기 위한 감정을 제외한 다른 라벨 제거
#     data_names_temp = []
#     data_paths_temp = []
#     labels_temp = []
#     genders_temp = []
#     texts_temp = []
#     for i in range(len(labels_one)):
#         if labels_one[i] in label_7_mapping.keys():
#             data_names_temp.append(data_names_one[i])
#             data_paths_temp.append(os.path.join(datasetname, 'audio', data_names_one[i]))
#             labels_temp.append(labels_one[i])
#             genders_temp.append(genders_one[i])
#             texts_temp.append(tests_one[i])

#     data_names = data_names + data_names_temp
#     data_paths = data_paths + data_paths_temp
#     labels = labels + labels_temp
#     genders = genders + genders_temp
#     texts = texts + texts_temp
    
# for datasetname in data_path_UTF_8:
#     ## labels.csv 파일을 dict 형식으로 저장
#     print(datasetname)
#     data_info = pd.read_csv(os.path.join(datasetname, "labels.csv"), encoding = "UTF-8").to_dict(orient = 'list')

#     ## 데이터 이름과 해당 데이터의 라벨값 저장
#     data_names_one = data_info["audio"]
#     labels_one = data_info["label"]
#     genders_one = data_info["gender"]
#     tests_one = data_info["text"]

#     ## 학습하기 위한 감정을 제외한 다른 라벨 제거
#     data_names_temp = []
#     data_paths_temp = []
#     labels_temp = []
#     genders_temp = []
#     texts_temp = []
#     for i in range(len(labels_one)):
#         if labels_one[i] in label_7_mapping.keys():
#             data_names_temp.append(data_names_one[i])
#             data_paths_temp.append(os.path.join(datasetname, 'audio', data_names_one[i]))
#             labels_temp.append(labels_one[i])
#             genders_temp.append(genders_one[i])
#             texts_temp.append(tests_one[i])

#     data_names = data_names + data_names_temp
#     data_paths = data_paths + data_paths_temp
#     labels = labels + labels_temp
#     genders = genders + genders_temp
#     texts = texts + texts_temp

# print('***  data loading Done  ***')
# print()

# print('***  data selecting (1) ....  ***')

# label_7_conut = {
#     "ang": 0,   # 분노
#     "dis": 0,   # 싫음
#     "fea": 0,   # 두려움
#     "hap": 0,   # 행복
#     "sad": 0,   # 슬픔
#     "sur": 0,   # 놀람
#     "neu": 0,   # 중립
# }

# labels_order = ["ang", "dis", "fea", "hap", "sad", "sur", "neu"]
# emotion_valid_data_name = [[], [], [], [], [], [], []]
# emotion_valid_data_path = [[], [], [], [], [], [], []]
# emotion_valid_data_label = [[], [], [], [], [], [], []]
# emotion_valid_data_gender = [[], [], [], [], [], [], []]
# emotion_valid_data_text = [[], [], [], [], [], [], []]

# iterations = tqdm(range(len(data_paths)))

# for i in iterations:
    
#     if labels[i] in list(label_7_mapping.keys()):
        
#         wave_form, sample_rate = torchaudio.load(data_paths[i], normalize=True)
#         audio_len = wave_form.size(1)
        
#         if audio_len >= sample_rate and audio_len < sample_rate * 10:
            
#             label_7_conut[labels[i]] += 1
#             emotion_valid_data_name[label_7_mapping[labels[i]]].append(data_names[i])
#             emotion_valid_data_path[label_7_mapping[labels[i]]].append(data_paths[i])
#             emotion_valid_data_label[label_7_mapping[labels[i]]].append(labels[i])
#             emotion_valid_data_gender[label_7_mapping[labels[i]]].append(genders[i])
#             emotion_valid_data_text[label_7_mapping[labels[i]]].append(texts[i])

# print(label_7_conut)
# print('***  data selecting (1) Done  ***')
# print()

# print('***  data selecting (2) ....  ***')

# data_dcit = {}

# for i in range(7):
#     nums = len(emotion_valid_data_label[i])
#     index_list = [i for i in range(nums)]
#     rand_index = random.sample(index_list, 14000)
    
#     iterations = tqdm(range(14000))
#     for j in iterations:
        
#         nums_14000 = rand_index[j]
#         if j < 10000:
#             data_one_info = [emotion_valid_data_label[i][nums_14000], emotion_valid_data_gender[i][nums_14000], emotion_valid_data_text[i][nums_14000], "train"]
#         elif j < 12000:
#             data_one_info = [emotion_valid_data_label[i][nums_14000], emotion_valid_data_gender[i][nums_14000], emotion_valid_data_text[i][nums_14000], "valid"]
#         else:
#             data_one_info = [emotion_valid_data_label[i][nums_14000], emotion_valid_data_gender[i][nums_14000], emotion_valid_data_text[i][nums_14000], "test"]
        
#         data_dcit[emotion_valid_data_name[i][nums_14000]] = data_one_info
        
#         shutil.copyfile(emotion_valid_data_path[i][nums_14000], './TrVTe_dataset/audio/' + emotion_valid_data_name[i][nums_14000])

# print('***  data selecting (2) Done  ***')
# print()

# print('***  data saving ....  ***')

# column = ['label', 'gender', 'text', 'TrVTe']
# df = pd.DataFrame.from_dict(data = data_dcit, orient = 'index', columns = column)
# df.index.name = 'audio'
# df.to_csv('./TrVTe_dataset/labels.csv', encoding = 'UTF-8')

# print('***  data saving Done  ***')
# print()



print('***  data zipping ....  ***')

zip_file = zipfile.ZipFile("./TrVTe_dataset/audio.zip", "w")  # "w": write 모드
for file in os.listdir('./TrVTe_dataset/audio'):
    if file.endswith('.wav'):
        zip_file.write(os.path.join('./TrVTe_dataset/audio', file), compress_type=zipfile.ZIP_DEFLATED)

zip_file.close()

print('***  data zipping Done  ***')
print()


print('***  data uploading ....  ***')

from minio import Minio # 7.1.14
from minio.commonconfig import CopySource

minioClient = Minio(endpoint="192.168.2.139:6731",
                    access_key="hee",
                    secret_key="!Wkdal8d192b@",
                    secure=False)

minioClient.fput_object(bucket_name="aaai-emotion", # 파일을 저장한 버킷 선택
                        object_name="audio/TrVTe_dataset/labels.csv", # 버킷내 파일을 저장 할 경로
                        file_path="./TrVTe_dataset/labels.csv" # 올리고 싶은 파일
                        )

minioClient.fput_object(bucket_name="aaai-emotion", # 파일을 저장한 버킷 선택
                        object_name="audio/TrVTe_dataset/audio.zip", # 버킷내 파일을 저장 할 경로
                        file_path="./TrVTe_dataset/audio.zip" # 올리고 싶은 파일
                        )

print('***  data uploading Done  ***')
print()