import os
import json
import pandas as pd
from tqdm.auto import tqdm

emotion_dict = {}


with open('./training_label.json', 'rt', encoding = 'UTF8') as json_file:
    label_data = json.load(json_file)
with open('./validation_label.json', 'rt', encoding = 'UTF8') as json_file:
    label_data = label_data + json.load(json_file)
    
emotion_mapping = {'분노': 'ang', 
             '당황': 'emb', 
             '슬픔': 'sad', 
             '무감정': 'neu', 
             '상처': 'sad', 
             '기쁨': 'hap',
             '불안': 'fea', 
             }

data_dcit = {}
column = ['label', 'gender', 'text', 'age', 'kind']

iterations = tqdm(label_data)

for data_info in iterations:
    
    age = data_info['reciter']['age']
    gender = data_info['reciter']['gender'][0]
    if gender == 'F':
        gender = 'female'
    elif gender == 'M':
        gender = 'male'
    else:
        gender = ''
    kind = data_info['recite_src']['literature']['genre']
    emotion = emotion_mapping[data_info['recite_src']['styles'][0]['emotion']]
    
    
    
    for data in data_info['sentences']:
        text = data['origin_text']
        file_name = data['voice_piece']['filename'][-18:-4]
        
        data_dcit[file_name] = [emotion, gender, text, age, kind]
    

        
df = pd.DataFrame.from_dict(data = data_dcit, orient = 'index', columns = column)
df.index.name = 'audio'
df.to_csv('./labels.csv', encoding = 'cp949')
        


