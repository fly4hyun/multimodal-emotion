




import os
import json
import pandas as pd
import shutil
from tqdm.auto import tqdm


emotion_mapping = {'Angry': 'ang', 
             'Embarrassed': 'emb', 
             'Sad': 'sad', 
             'Neutrality': 'neu', 
             'Hurt': 'sad', 
             'Happy': 'hap',
             'Anxious': 'fea', 
             'N/A': '', 
             }


data_dcit = {}
column = ['label', 'gender', 'text', 'age']

data_folder_list = os.listdir('../emotion_and_speaker_style/audio')
asdf = []
iterations = tqdm(data_folder_list)

for data_name in iterations:


    if os.path.exists('../emotion_and_speaker_style/labels/' + data_name[:-11] + '/' + data_name[:-4] + '.json'):
        with open('../emotion_and_speaker_style/labels/' + data_name[:-11] + '/' + data_name[:-4] + '.json', 'rt', encoding = 'UTF8') as json_file:
            try:
                data_info = json.load(json_file)
            except:
                print(data_name)
            
        label = data_info['화자정보']['Emotion']
        label = emotion_mapping[label]
        gender = data_info['화자정보']['Gender'].lower()
        text = data_info['전사정보']['TransLabelText']
        age = int(data_info['화자정보']['Age'][:2])
        
        data_dcit[data_name] = [label, gender, text, age]
        

            
df = pd.DataFrame.from_dict(data = data_dcit, orient = 'index', columns = column)
df.index.name = 'audio'
df.to_csv('./labels.csv', encoding = 'UTF-8')
            



















