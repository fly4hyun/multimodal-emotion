
import os
import shutil
import pandas as pd
import random
import json

from sklearn.model_selection import train_test_split

"""
1~50    3
51~100  5
101~150 6
151~200 2
201~250 1
251~300 0
301~350 4
"""

###################################################################################################

emotion_mapping_34 = {
    
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

###################################################################################################

data_path = "./data"
labels_mapping_csv = train_csv = pd.read_csv(os.path.join(data_path, 'labels_mapping_34.csv'), encoding = 'UTF-8').to_dict(orient = 'list')

multi_data_names = train_csv['audio']
multi_data_labels_ed = train_csv['multi_emotion']

multi_datas = {}

for multi_data_label_index in range(len(multi_data_labels_ed)):
    multi_data_label = multi_data_labels_ed[multi_data_label_index]
    label_temp = []
    label_split = multi_data_label.split(" ")
    for label_one in label_split:
        label_temp.append(emotion_mapping_34[label_one])
    
    multi_datas[multi_data_names[multi_data_label_index]] = label_temp
    
###################################################################################################

with open('./file_num_to_text_num.json', 'r') as f:

    json_data = json.load(f)

###################################################################################################

data_path = "./data"
new_data_path = "./data_new"
train_csv = pd.read_csv(os.path.join(data_path, 'train_labels.csv'), encoding = 'UTF-8').to_dict(orient = 'list')
test_csv = pd.read_csv(os.path.join(data_path, 'test_labels.csv'), encoding = 'UTF-8').to_dict(orient = 'list')

train_data_names = train_csv['audio']
train_data_labels = train_csv['label']

train_data_names, valid_data_names, train_data_labels, valid_data_labels = train_test_split(train_data_names, train_data_labels, test_size = 1/10, random_state = 444)


test_data_names = test_csv['audio']
test_data_labels = test_csv['label']

data_names = train_data_names + valid_data_names + test_data_names
data_labels = train_data_labels + valid_data_labels + test_data_labels

train_data_names_new = []
train_data_labels_new = []
valid_data_names_new = []
valid_data_labels_new = []
test_data_names_new = []
test_data_labels_new = []

###################################################################################################

for i_index in range(len(train_data_names)):
    data_name = train_data_names[i_index]
    if data_name[:6] != 'studio':
        train_data_names_new.append(data_name)
        train_data_labels_new.append(train_data_labels[i_index])
        
        shutil.copyfile(os.path.join(data_path, "IE_features", "train", data_name[:-3] + "pt"), 
                        os.path.join(new_data_path, "IE_features", "train", data_name[:-3] + "pt"))
        shutil.copyfile(os.path.join(data_path, "SE_features", "train", data_name[:-3] + "pt"), 
                        os.path.join(new_data_path, "SE_features", "train", data_name[:-3] + "pt"))
        shutil.copyfile(os.path.join(data_path, "TE_features", "train", data_name[:-3] + "pt"), 
                        os.path.join(new_data_path, "TE_features", "train", data_name[:-3] + "pt"))

for i_index in range(len(valid_data_names)):
    data_name = valid_data_names[i_index]
    if data_name[:6] != 'studio':
        valid_data_names_new.append(data_name)
        valid_data_labels_new.append(valid_data_labels[i_index])
        
        shutil.copyfile(os.path.join(data_path, "IE_features", "train", data_name[:-3] + "pt"), 
                        os.path.join(new_data_path, "IE_features", "valid", data_name[:-3] + "pt"))
        shutil.copyfile(os.path.join(data_path, "SE_features", "train", data_name[:-3] + "pt"), 
                        os.path.join(new_data_path, "SE_features", "valid", data_name[:-3] + "pt"))
        shutil.copyfile(os.path.join(data_path, "TE_features", "train", data_name[:-3] + "pt"), 
                        os.path.join(new_data_path, "TE_features", "valid", data_name[:-3] + "pt"))

for i_index in range(len(test_data_names)):
    data_name = test_data_names[i_index]
    if data_name[:6] != 'studio':
        test_data_names_new.append(data_name)
        test_data_labels_new.append(test_data_labels[i_index])
        
        shutil.copyfile(os.path.join(data_path, "IE_features", "test", data_name[:-3] + "pt"), 
                        os.path.join(new_data_path, "IE_features", "test", data_name[:-3] + "pt"))
        shutil.copyfile(os.path.join(data_path, "SE_features", "test", data_name[:-3] + "pt"), 
                        os.path.join(new_data_path, "SE_features", "test", data_name[:-3] + "pt"))
        shutil.copyfile(os.path.join(data_path, "TE_features", "test", data_name[:-3] + "pt"), 
                        os.path.join(new_data_path, "TE_features", "test", data_name[:-3] + "pt"))
        
###################################################################################################

data_names_new = []
data_labels_new = []

for i_index in range(len(data_names)):
    data_name = data_names[i_index]
    if data_name[:6] == 'studio':
        data_names_new.append(data_names[i_index])
        data_labels_new.append(data_labels[i_index])

###################################################################################################

studio_train_data = []
studio_train_label = []
studio_valid_data = []
studio_valid_label = []
studio_test_data = []
studio_test_label = []

###################################################################################################

while 1:
    train_multi_count = [0 for _ in range(34)]
    valid_multi_count = [0 for _ in range(34)]
    test_multi_count = [0 for _ in range(34)]
    
    train_data = {}
    valid_data = {}
    test_data = {}
    
    new_mapping_list = []
    new_valid_mapping_list = []
    
    for i_index in range(7):
        one_to_50_list = [i + 1 for i in range(50)]
        
        ### test
        
        x = random.randint(1, 50)
        new_mapping_list.append(one_to_50_list[x - 1] + 50 * i_index)
        del one_to_50_list[x - 1]
        
        x = random.randint(1, 49)
        new_mapping_list.append(one_to_50_list[x - 1] + 50 * i_index)
        del one_to_50_list[x - 1]
        
        x = random.randint(1, 48)
        new_mapping_list.append(one_to_50_list[x - 1] + 50 * i_index)
        del one_to_50_list[x - 1]
        
        x = random.randint(1, 47)
        new_mapping_list.append(one_to_50_list[x - 1] + 50 * i_index)
        del one_to_50_list[x - 1]
        
        x = random.randint(1, 46)
        new_mapping_list.append(one_to_50_list[x - 1] + 50 * i_index)
        del one_to_50_list[x - 1]
        
        ### valid
        
        x = random.randint(1, 45)
        new_valid_mapping_list.append(one_to_50_list[x - 1] + 50 * i_index)
        del one_to_50_list[x - 1]
        
        x = random.randint(1, 44)
        new_valid_mapping_list.append(one_to_50_list[x - 1] + 50 * i_index)
        del one_to_50_list[x - 1]
        
        x = random.randint(1, 43)
        new_valid_mapping_list.append(one_to_50_list[x - 1] + 50 * i_index)
        del one_to_50_list[x - 1]
        
        x = random.randint(1, 42)
        new_valid_mapping_list.append(one_to_50_list[x - 1] + 50 * i_index)
        del one_to_50_list[x - 1]
        
        x = random.randint(1, 41)
        new_valid_mapping_list.append(one_to_50_list[x - 1] + 50 * i_index)
        del one_to_50_list[x - 1]
    
    for i_index in range(len(data_names_new)):
        num_index = data_names_new[i_index][-9:-4]
        
        if int(json_data[num_index]) in new_mapping_list:
            studio_test_data.append(data_names_new[i_index])
            studio_test_label.append(data_labels_new[i_index])
            test_data[data_names_new[i_index]] = [data_labels_new[i_index]]
            
        elif int(json_data[num_index]) in new_valid_mapping_list:
            studio_valid_data.append(data_names_new[i_index])
            studio_valid_label.append(data_labels_new[i_index])
            valid_data[data_names_new[i_index]] = [data_labels_new[i_index]]
            
        else:
            studio_train_data.append(data_names_new[i_index])
            studio_train_label.append(data_labels_new[i_index])
            train_data[data_names_new[i_index]] = [data_labels_new[i_index]]

    for i_index in range(len(studio_train_data)):
        for multi_label in multi_datas[studio_train_data[i_index]]:
            train_multi_count[multi_label] += 1
                    
    for i_index in range(len(studio_valid_data)):
        for multi_label in multi_datas[studio_valid_data[i_index]]:
            valid_multi_count[multi_label] += 1
            
    for i_index in range(len(studio_test_data)):
        for multi_label in multi_datas[studio_test_data[i_index]]:
            test_multi_count[multi_label] += 1
    
    print()    
    print('train : ', train_multi_count)
    print('valid : ', valid_multi_count)
    print('test  : ', test_multi_count)

    if (0 not in  train_multi_count) and (0 not in valid_multi_count) and (0 not in test_multi_count):
        break

for data_name in studio_train_data:

    if os.path.exists(os.path.join(data_path, "IE_features", "train", data_name[:-3] + "pt")):
        shutil.copyfile(os.path.join(data_path, "IE_features", "train", data_name[:-3] + "pt"), 
                        os.path.join(new_data_path, "IE_features", "train", data_name[:-3] + "pt"))
    else:
        shutil.copyfile(os.path.join(data_path, "IE_features", "test", data_name[:-3] + "pt"), 
                        os.path.join(new_data_path, "IE_features", "train", data_name[:-3] + "pt"))
        
    if os.path.exists(os.path.join(data_path, "SE_features", "train", data_name[:-3] + "pt")):
        shutil.copyfile(os.path.join(data_path, "SE_features", "train", data_name[:-3] + "pt"), 
                        os.path.join(new_data_path, "SE_features", "train", data_name[:-3] + "pt"))
    else:
        shutil.copyfile(os.path.join(data_path, "SE_features", "test", data_name[:-3] + "pt"), 
                        os.path.join(new_data_path, "SE_features", "train", data_name[:-3] + "pt"))
        
    if os.path.exists(os.path.join(data_path, "TE_features", "train", data_name[:-3] + "pt")):
        shutil.copyfile(os.path.join(data_path, "TE_features", "train", data_name[:-3] + "pt"), 
                        os.path.join(new_data_path, "TE_features", "train", data_name[:-3] + "pt"))
    else:
        shutil.copyfile(os.path.join(data_path, "TE_features", "test", data_name[:-3] + "pt"), 
                        os.path.join(new_data_path, "TE_features", "train", data_name[:-3] + "pt"))

for data_name in studio_valid_data:

    if os.path.exists(os.path.join(data_path, "IE_features", "train", data_name[:-3] + "pt")):
        shutil.copyfile(os.path.join(data_path, "IE_features", "train", data_name[:-3] + "pt"), 
                        os.path.join(new_data_path, "IE_features", "valid", data_name[:-3] + "pt"))
    else:
        shutil.copyfile(os.path.join(data_path, "IE_features", "test", data_name[:-3] + "pt"), 
                        os.path.join(new_data_path, "IE_features", "valid", data_name[:-3] + "pt"))
        
    if os.path.exists(os.path.join(data_path, "SE_features", "train", data_name[:-3] + "pt")):
        shutil.copyfile(os.path.join(data_path, "SE_features", "train", data_name[:-3] + "pt"), 
                        os.path.join(new_data_path, "SE_features", "valid", data_name[:-3] + "pt"))
    else:
        shutil.copyfile(os.path.join(data_path, "SE_features", "test", data_name[:-3] + "pt"), 
                        os.path.join(new_data_path, "SE_features", "valid", data_name[:-3] + "pt"))
        
    if os.path.exists(os.path.join(data_path, "TE_features", "train", data_name[:-3] + "pt")):
        shutil.copyfile(os.path.join(data_path, "TE_features", "train", data_name[:-3] + "pt"), 
                        os.path.join(new_data_path, "TE_features", "valid", data_name[:-3] + "pt"))
    else:
        shutil.copyfile(os.path.join(data_path, "TE_features", "test", data_name[:-3] + "pt"), 
                        os.path.join(new_data_path, "TE_features", "valid", data_name[:-3] + "pt"))
        
for data_name in studio_test_data:

    if os.path.exists(os.path.join(data_path, "IE_features", "train", data_name[:-3] + "pt")):
        shutil.copyfile(os.path.join(data_path, "IE_features", "train", data_name[:-3] + "pt"), 
                        os.path.join(new_data_path, "IE_features", "test", data_name[:-3] + "pt"))
    else:
        shutil.copyfile(os.path.join(data_path, "IE_features", "test", data_name[:-3] + "pt"), 
                        os.path.join(new_data_path, "IE_features", "test", data_name[:-3] + "pt"))
        
    if os.path.exists(os.path.join(data_path, "SE_features", "train", data_name[:-3] + "pt")):
        shutil.copyfile(os.path.join(data_path, "SE_features", "train", data_name[:-3] + "pt"), 
                        os.path.join(new_data_path, "SE_features", "test", data_name[:-3] + "pt"))
    else:
        shutil.copyfile(os.path.join(data_path, "SE_features", "test", data_name[:-3] + "pt"), 
                        os.path.join(new_data_path, "SE_features", "test", data_name[:-3] + "pt"))
        
    if os.path.exists(os.path.join(data_path, "TE_features", "train", data_name[:-3] + "pt")):
        shutil.copyfile(os.path.join(data_path, "TE_features", "train", data_name[:-3] + "pt"), 
                        os.path.join(new_data_path, "TE_features", "test", data_name[:-3] + "pt"))
    else:
        shutil.copyfile(os.path.join(data_path, "TE_features", "test", data_name[:-3] + "pt"), 
                        os.path.join(new_data_path, "TE_features", "test", data_name[:-3] + "pt"))

for i_index in range(len(train_data_names_new)):
    train_data[train_data_names_new[i_index]] = [train_data_labels_new[i_index]]
for i_index in range(len(valid_data_names_new)):
    valid_data[valid_data_names_new[i_index]] = [valid_data_labels_new[i_index]]
for i_index in range(len(test_data_names_new)):
    test_data[test_data_names_new[i_index]] = [test_data_labels_new[i_index]]

columns = ['label']

df = pd.DataFrame.from_dict(data = train_data, orient = 'index', columns = columns)
df.index.name = 'audio'
df.to_csv('./data_new/train_labels.csv', encoding = 'cp949')

df_ = pd.DataFrame.from_dict(data = valid_data, orient = 'index', columns = columns)
df_.index.name = 'audio'
df_.to_csv('./data_new/valid_labels.csv', encoding = 'cp949')

df__ = pd.DataFrame.from_dict(data = test_data, orient = 'index', columns = columns)
df__.index.name = 'audio'
df__.to_csv('./data_new/test_labels.csv', encoding = 'cp949')

