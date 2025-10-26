###################################################################################################

import warnings
warnings.filterwarnings(action='ignore')

import pandas as pd
import os
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import Dataset
import torch.nn.functional as F

###################################################################################################
#
#   TE : 72.28
#   TE : 15.00
#
#   SE : 93.114
#   SE : 22.78
#
#   SE : 20.37 (flagship)
#   SE : 19.78 (multimodal_data)
#   SE : 20.22 (multimodal video)
#   SE : 34.78 (studio)
#
###################################################################################################

# {

#     '분노' : 'ang', '억울함': 'unf', 
#     '싫음': 'dis', '지루함': 'dul', '미워함': 'hat', '짜증남': 'irr',
#     '두려움': 'fea', '창피함': 'ash', '불안': 'anx', '당황': 'emb',
#     '행복': 'hap', '그리움': 'mis', '기쁨': 'ple', '사랑': 'lov', '황홀함': 'mag', '감동': 'mov',
#     '즐거움': joy, '홀가분함': 'lit', '설렘': 'rom', '만족': 'stf', '자신감': 'cfd', '고마움': "thx",
#     '슬픔': 'sad', '미안함': 'sor', '안타까움': 'reg', '외로움': 'lon', '실망': 'dip',
#     '괴로움': 'pai', '부러움': 'env', '후회': 'reg',
#     '놀람': 'sur',
#     '중립': 'neu', '바람': 'wis', '무관심': 'ind',

#     '경멸' : 'con'

# }

# emotion_mapping_34 = {
    
#     'ang': 0, 'unf': 1,

#     'dis': 2, 'dul': 3, 'hat': 4, 'irr': 5,

#     'fea': 6, 'ash': 7, 'anx': 8, 'emb': 9,

#     'hap': 10, 'mis': 11, 'ple': 12, 'lov': 13, 'mag': 14, 'mov': 15,
#     'joy': 16, 'lit': 17, 'rom': 18, 'stf': 19, 'cfd': 20, 'thx': 21,

#     'sad': 22, 'sor': 23, 'reg': 24, 'lon': 25, 'dip': 26,
#     'pai': 27, 'env': 28, 'reg': 29,

#     'sur': 30,

#     'neu': 31, 'wis': 32, 'ind': 33
# }

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

num_classes = 34

###################################################################################################

label_7_mapping = {
    "ang": 0,   # 분노
    "dis": 1,   # 싫음
    "fea": 2,   # 두려움
    "hap": 3,   # 행복
    "sad": 4,   # 슬픔
    "sur": 5,   # 놀람
    "neu": 6,   # 중립
}

num_uniemotion = 7

###################################################################################################

def multi_hot_encoding(labels):
    
    encode = [1. if i in labels else 0. for i in range(num_classes)]
    
    return encode


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

data_path = "./data"
train_csv = pd.read_csv(os.path.join(data_path, 'train_labels.csv'), encoding = 'UTF-8').to_dict(orient = 'list')

train_data_names = train_csv['audio']
train_data_uni_labels = train_csv['label']
train_data_indexes = [i for i in range(len(train_data_names))]

train_data_labels = []
for train_data in train_data_names:
    train_data_labels.append(multi_datas[train_data])

#train_data_indexes, valid_data_indexes, train_data_labels, valid_data_labels = train_test_split(train_data_indexes, train_data_labels, test_size = 1/10, random_state = 444)

train_ie_feature_paths = []
train_se_feature_paths = []
train_te_feature_paths = []
train_uni_labels = []

for data_index in train_data_indexes:
    data_name = train_data_names[data_index]
    train_ie_feature_paths.append(os.path.join(data_path, "IE_features", "train", data_name[:-3] + "pt"))
    train_se_feature_paths.append(os.path.join(data_path, "SE_features", "train", data_name[:-3] + "pt"))
    train_te_feature_paths.append(os.path.join(data_path, "TE_features", "train", data_name[:-3] + "pt"))
    train_uni_labels.append(train_data_uni_labels[data_index])

###################################################################################################

data_path = "./data"
valid_csv = pd.read_csv(os.path.join(data_path, 'valid_labels.csv'), encoding = 'UTF-8').to_dict(orient = 'list')

valid_data_names = valid_csv['audio']
valid_data_uni_labels = valid_csv['label']
valid_data_indexes = [i for i in range(len(valid_data_names))]

valid_data_labels = []
for valid_data in valid_data_names:
    valid_data_labels.append(multi_datas[valid_data])

valid_ie_feature_paths = []
valid_se_feature_paths = []
valid_te_feature_paths = []
valid_uni_labels = []

for data_index in valid_data_indexes:
    data_name = valid_data_names[data_index]
    valid_ie_feature_paths.append(os.path.join(data_path, "IE_features", "valid", data_name[:-3] + "pt"))
    valid_se_feature_paths.append(os.path.join(data_path, "SE_features", "valid", data_name[:-3] + "pt"))
    valid_te_feature_paths.append(os.path.join(data_path, "TE_features", "valid", data_name[:-3] + "pt"))
    valid_uni_labels.append(valid_data_uni_labels[data_index])

###################################################################################################

data_path = "./data"
test_csv = pd.read_csv(os.path.join(data_path, 'test_labels.csv'), encoding = 'UTF-8').to_dict(orient = 'list')

test_data_names = test_csv['audio']
test_data_uni_labels = test_csv['label']
test_data_indexes = [i for i in range(len(test_data_names))]

test_data_labels = []
for test_data in test_data_names:
    test_data_labels.append(multi_datas[test_data])

test_ie_feature_paths = []
test_se_feature_paths = []
test_te_feature_paths = []
test_uni_labels = []

for data_index in test_data_indexes:
    data_name = test_data_names[data_index]
    test_ie_feature_paths.append(os.path.join(data_path, "IE_features", "test", data_name[:-3] + "pt"))
    test_se_feature_paths.append(os.path.join(data_path, "SE_features", "test", data_name[:-3] + "pt"))
    test_te_feature_paths.append(os.path.join(data_path, "TE_features", "test", data_name[:-3] + "pt"))
    test_uni_labels.append(test_data_uni_labels[data_index])

###################################################################################################

class CustomDataset(Dataset):
    def __init__(self, data_type = "train"):
        
        if data_type == "train":
            self.iefs = train_ie_feature_paths
            self.sefs = train_se_feature_paths
            self.tefs = train_te_feature_paths
            self.labels = train_data_labels
            self.uni_labels = train_uni_labels
        elif data_type == "valid":
            self.sefs = valid_se_feature_paths
            self.iefs = valid_ie_feature_paths
            self.tefs = valid_te_feature_paths
            self.labels = valid_data_labels
            self.uni_labels = valid_uni_labels
        elif data_type == "test":
            self.sefs = test_se_feature_paths
            self.iefs = test_ie_feature_paths
            self.tefs = test_te_feature_paths
            self.labels = test_data_labels
            self.uni_labels = test_uni_labels
            
    def __len__(self):
        
        return len(self.labels)
    
    def __getitem__(self, idx):
        
        ief = torch.load(self.iefs[idx]).squeeze(0)    ### 1536
        sef = torch.load(self.sefs[idx]).squeeze(0)    ### 1024
        tef = torch.load(self.tefs[idx]).squeeze(0)    ### 768
        
        label = torch.tensor(multi_hot_encoding(self.labels[idx]))
        uni_label = F.one_hot(torch.tensor(label_7_mapping[self.uni_labels[idx]]), num_classes = num_uniemotion)
        
        data = {
            'ie': ief, 
            'se': sef, 
            'te': tef,
            'label': label,
            'uni_label': uni_label
        }
        
        return data

###################################################################################################

trainset = CustomDataset('train')
validset = CustomDataset('valid')
testset = CustomDataset('test')

###################################################################################################

if __name__ == '__main__':
    
    from torch.utils.data import DataLoader
    
    train_loader = DataLoader(trainset, batch_size = 1, shuffle = False, num_workers = 3, pin_memory = True)
    for train_data in train_loader:
        
        print(train_data['se'])
        
        break




