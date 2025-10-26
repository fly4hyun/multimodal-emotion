###################################################################################################

import warnings
warnings.filterwarnings(action='ignore')

import pandas as pd
import os
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import Dataset


# sef = torch.load("./data/SE_features/train/clip_1_131.pt")

# print(sef.shape)


data_path = "./data"
train_csv = pd.read_csv(os.path.join(data_path, 'train_labels.csv'), encoding = 'UTF-8').to_dict(orient = 'list')

train_data_names = train_csv['audio']
train_data_labels = train_csv['label']

train_data_names.sort()

train_se_feature_paths = []
train_te_feature_paths = []
train_se = 0
train_te = 0
valid_se = 0
valid_te = 0
test_se = 0
test_te = 0

train_num = 0
test_num = 0
for data_name in train_data_names:
    sef = os.path.join(data_path, "SE_features", "train", data_name[:-3] + "pt")
    tef = os.path.join(data_path, "TE_features", "train", data_name[:-3] + "pt")
    train_num = train_num + 1
    if not os.path.isfile(sef):
        print(sef)
        train_se = train_se + 1
    if not os.path.isfile(tef):
        print(tef)
        train_te = train_te + 1

valid_se_feature_paths = []
valid_te_feature_paths = []

###################################################################################################

data_path = "./data"
test_csv = pd.read_csv(os.path.join(data_path, 'test_labels.csv'), encoding = 'UTF-8').to_dict(orient = 'list')

test_data_names = test_csv['audio']
test_data_labels = test_csv['label']

test_se_feature_paths = []
test_te_feature_paths = []

for data_name in test_data_names:
    sef = os.path.join(data_path, "SE_features", "test", data_name[:-3] + "pt")
    tef = os.path.join(data_path, "TE_features", "test", data_name[:-3] + "pt")
    test_num = test_num + 1
    if not os.path.isfile(sef):
        print(sef)
        test_se = test_se + 1
    if not os.path.isfile(tef):
        print(tef)
        test_te = test_te + 1
        
# se_list = os.listdir("./data/SE_features/train")
# te_list = os.listdir("./data/TE_features/train")
# se_list.sort()
# asd = 0
# for i in se_list:
#     if i in te_list:
#         continue
#     else:
#         asd = asd + 1
#         print(i)
# print(asd)
        
print('train: ', train_num)
print('test: ', test_num)

print('se train ', train_se)
print('te train ', train_te)
print('se test ', test_se)
print('te test ', test_te)

tse = len(os.listdir("./data/SE_features/train"))
tte = len(os.listdir("./data/TE_features/train"))


print('se :', tse)
print('te :', tte)

tse = len(os.listdir("./data/SE_features/test"))
tte = len(os.listdir("./data/TE_features/test"))


print('se :', tse)
print('te :', tte)


