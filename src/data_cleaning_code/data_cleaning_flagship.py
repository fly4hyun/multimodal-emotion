







import os
import json
import pandas as pd
from tqdm.auto import tqdm

emotion_mapping_34__ = {
    
    '분노': 0, '억울함': 1,

    '싫음': 2, '지루함': 3, '미워함': 4, '짜증남': 5,

    '두려움': 6, '창피함': 7, '불안': 8, '당황': 9,

    '행복': 10, '그리움': 11, '기쁨': 12, '사랑': 13, '황홀함': 14, '감동': 15,
    '즐거움': 16, '홀가분함': 17, '설렘': 18, '만족': 19, '자신감': 20, '고마움': 21,

    '슬픔': 22, '미안함': 23, '안타까움': 24, '외로움': 25, '실망': 26,
    '괴로움': 27, '부러움': 28, '후회': 29,

    '놀람': 30,

    '중립': 31, '바람': 32, '무관심': 33
}

emotion_mapping_34 = {

    '0' : 'ang', '1': 'unf',
    '2': 'dis', '3': 'dul', '4': 'hat', '5': 'irr',
    '6': 'fea', '7': 'ash', '8': 'anx', '9': 'emb',
    '10': 'hap', '11': 'mis', '12': 'ple', '13': 'lov', '14': 'mag', '15': 'mov',
    '16': 'joy', '17': 'lit', '18': 'rom', '19': 'stf', '20': 'cfd', '21': "thx",
    '22': 'sad', '23': 'sor', '24': 'reg', '25': 'lon', '26': 'dip',
    '27': 'pai', '28': 'env', '29': 'reg',
    '30': 'sur',
    '31': 'neu', '32': 'wis', '33': 'ind',

    '34' : 'con'

}

emotion_mapping_7 = {
    
    '분노': 0,

    '싫음': 1,

    '두려움': 2,

    '행복': 3,

    '슬픔': 4,

    '놀람': 5,

    '중립': 6,
}

emotion_mapping = {0: 'ang', 
             1: 'dis', 
             2: 'fea', 
             3: 'hap', 
             4: 'sad', 
             5: 'sur',
             6: 'neu', 
             }

column = ['label', 'gender', 'text', 'multi_emotion']
data_dcit = {}

with open('./labels.json', 'rt', encoding = 'UTF8') as json_file:
    label_data = json.load(json_file)


key_list = label_data.keys()
for keys in key_list:
    
    name = 'flagship_a_' + keys
    label = emotion_mapping[label_data[keys]['emotion']]
    text = label_data[keys]['text']
    gender = ''
    
    multi_emotion = label_data[keys]['multi_emotion']
    mul_emo = ''
    for iii in range(len(multi_emotion)):
        if iii == 0:
            mul_emo = mul_emo + emotion_mapping_34[str(multi_emotion[iii])]
        else:
            mul_emo = mul_emo + ' ' + emotion_mapping_34[str(multi_emotion[iii])]
    
    
    data_dcit[name] = [label, gender, text, mul_emo]
    
    

    
    
df = pd.DataFrame.from_dict(data = data_dcit, orient = 'index', columns = column)
df.index.name = 'audio'
df.to_csv('./labels.csv', encoding = 'cp949')















