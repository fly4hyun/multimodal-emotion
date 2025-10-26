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

label_7_conut = {
    "ang": 0,   # 분노
    "dis": 0,   # 싫음
    "fea": 0,   # 두려움
    "hap": 0,   # 행복
    "sad": 0,   # 슬픔
    "sur": 0,   # 놀람
    "neu": 0,   # 중립
}

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

num_classes = 7

###################################################################################################

data_path = "./data"
train_csv = pd.read_csv(os.path.join(data_path, 'train_labels.csv'), encoding = 'UTF-8').to_dict(orient = 'list')

train_data_names = train_csv['audio']
train_data_labels = train_csv['label']

#train_data_names, valid_data_names, train_data_labels, valid_data_labels = train_test_split(train_data_names, train_data_labels, test_size = 1/10, random_state = 444)

train_ie_feature_paths = []
train_se_feature_paths = []
train_te_feature_paths = []

for data_name in train_data_names:
    train_ie_feature_paths.append(os.path.join(data_path, "IE_features", "train", data_name[:-3] + "pt"))
    train_se_feature_paths.append(os.path.join(data_path, "SE_features", "train", data_name[:-3] + "pt"))
    train_te_feature_paths.append(os.path.join(data_path, "TE_features", "train", data_name[:-3] + "pt"))

###################################################################################################

data_path = "./data"
valid_csv = pd.read_csv(os.path.join(data_path, 'valid_labels.csv'), encoding = 'UTF-8').to_dict(orient = 'list')

valid_data_names = valid_csv['audio']
valid_data_labels = valid_csv['label']

valid_ie_feature_paths = []
valid_se_feature_paths = []
valid_te_feature_paths = []

for data_name in valid_data_names:
    valid_ie_feature_paths.append(os.path.join(data_path, "IE_features", "valid", data_name[:-3] + "pt"))
    valid_se_feature_paths.append(os.path.join(data_path, "SE_features", "valid", data_name[:-3] + "pt"))
    valid_te_feature_paths.append(os.path.join(data_path, "TE_features", "valid", data_name[:-3] + "pt"))

###################################################################################################

data_path = "./data"
test_csv = pd.read_csv(os.path.join(data_path, 'test_labels.csv'), encoding = 'UTF-8').to_dict(orient = 'list')

test_data_names = test_csv['audio']
test_data_labels = test_csv['label']

test_ie_feature_paths = []
test_se_feature_paths = []
test_te_feature_paths = []

for data_name in test_data_names:
    test_ie_feature_paths.append(os.path.join(data_path, "IE_features", "test", data_name[:-3] + "pt"))
    test_se_feature_paths.append(os.path.join(data_path, "SE_features", "test", data_name[:-3] + "pt"))
    test_te_feature_paths.append(os.path.join(data_path, "TE_features", "test", data_name[:-3] + "pt"))

###################################################################################################

class CustomDataset(Dataset):
    def __init__(self, data_type = "train"):
        
        if data_type == "train":
            self.iefs = train_ie_feature_paths
            self.sefs = train_se_feature_paths
            self.tefs = train_te_feature_paths
            self.labels = train_data_labels
        elif data_type == "valid":
            self.iefs = valid_ie_feature_paths
            self.sefs = valid_se_feature_paths
            self.tefs = valid_te_feature_paths
            self.labels = valid_data_labels
        elif data_type == "test":
            self.iefs = test_ie_feature_paths
            self.sefs = test_se_feature_paths
            self.tefs = test_te_feature_paths
            self.labels = test_data_labels
            
    def __len__(self):
        
        return len(self.labels)
    
    def __getitem__(self, idx):
        
        ief = torch.load(self.iefs[idx]).squeeze(0)    ### 1536
        sef = torch.load(self.sefs[idx]).squeeze(0)    ### 1024
        tef = torch.load(self.tefs[idx]).squeeze(0)    ### 768
        
        label = torch.tensor(label_7_mapping[self.labels[idx]])
        #label = F.one_hot(torch.tensor(label_7_mapping[self.labels[idx]]), num_classes = num_classes)
        #label = label[:6]

        data = {
            'ie': ief, 
            'se': sef, 
            'te': tef,
            'label': label,
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
        
        print(train_data['ie'])
        print(train_data['ie'].shape)
        
        break




