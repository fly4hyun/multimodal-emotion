




###################################################################################################

import warnings
warnings.filterwarnings(action='ignore')

import pandas as pd
import os
from tqdm import tqdm

import torch
import torch.nn.functional as F

import custom_model
from utils import ktop_result

###################################################################################################

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(234)
if device == 'cuda:0':
    torch.cuda.manual_seed_all(234)
print(device + ' is avaulable')

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

emotion_mapping_34_re = {
    
    0: '분노', 1: '억울함',

    2: '싫음', 3: '지루함', 4: '미워함', 5: '짜증남',

    6: '두려움', 7: '창피함', 8: '불안', 9: '당황',

    10: '행복', 11: '그리움', 12: '기쁨', 13: '사랑', 14: '황홀함', 15: '감동',
    16: '즐거움', 17: '홀가분함', 18:'설렘', 19: '만족', 20: '자신감', 21: '고마움',

    22: '슬픔', 23: '미안함', 24: '안타까움', 25: '외로움', 26: '실망',
    27: '괴로움', 28: '부러움', 29: '후회',

    30: '놀람',

    31: '중립', 32: '바람', 33: '무관심'
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

label_7_mapping_re = {
    0: "분노",
    1: "싫음",
    2: "두려움",
    3: "행복",
    4: "슬픔",
    5: "놀람",
    6: "중립",
}

num_uniemotion = 7

uni_emotion_masking = [0, 0, 
                       1, 1, 1, 1, 
                       2, 2, 2, 2, 
                       3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 
                       4, 4, 4, 4, 4, 4, 4, 4, 
                       5, 
                       6, 6, 6]
uni_emotion_masking = torch.LongTensor(uni_emotion_masking).to(device)

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

model_name_7 = 'Multi_Modal_TransformerEncoder3'
model_name_34 = "Multi_Modal_TransformerEncoder3"

model_path_7 = './models/' + model_name_7 + '.pth'
model_path_34 = './models_34/' + model_name_34 + '_top1.pth'

model_7 = custom_model.Multi_Modal_TransformerEncoder3(num_uniemotion, device)
model_7.load_state_dict(torch.load(model_path_7))
model_7.eval()
model_7 = model_7.to(device)

model_34 = custom_model.Multi_Modal_TransformerEncoder3(num_classes, device)
model_34.load_state_dict(torch.load(model_path_34))
model_34.eval()
model_34 = model_34.to(device)

###################################################################################################

data = {}

iterations = tqdm(range(len(test_data_names)))
# iterations = tqdm(zip(test_data_names, 
#                       test_ie_feature_paths, 
#                       test_se_feature_paths, 
#                       test_te_feature_paths, 
#                       test_data_labels, 
#                       test_uni_labels))
for num_i in iterations:
    
    name = test_data_names[num_i]
    ief_path = test_ie_feature_paths[num_i]
    sef_path = test_se_feature_paths[num_i]
    tef_path = test_te_feature_paths[num_i]
    multi_label = test_data_labels[num_i]
    uni_label = test_uni_labels[num_i]
    #name, ief_path, sef_path, tef_path, multi_label, uni_label = test_data_info[num_i]

    ief = torch.load(ief_path).to(device)
    sef = torch.load(sef_path).to(device)
    tef = torch.load(tef_path).unsqueeze(0).to(device)

    label = torch.LongTensor(multi_hot_encoding(multi_label)).to(device)
    uni_label = F.one_hot(torch.tensor(label_7_mapping[uni_label]), num_classes = num_uniemotion).to(device)
    
    output_7 = F.softmax(model_7(ief, sef, tef))
    output_34 = F.softmax(model_34(ief, sef, tef))
    
    ### 7 ###

    preds_7 = torch.max(output_7.data, 1)[1].item()
    p_7 = torch.max(output_7.data, 1)[0].item()
    target_7 = torch.max(uni_label.data, 0)[1].item()

    data_1_uni_label = label_7_mapping_re[target_7]
    data_3_uni_ox = 'O' if preds_7 == target_7 else 'X'
    data_7_uni_pred = p_7
    data_8_uni_7_p = label_7_mapping_re[preds_7]
    
    ### 34 ###    
    
    ox, multi_label_pred, pred, p, ox_7, pred_7 = ktop_result(output_34, label, 1, emotion_mapping_34_re, uni_label, label_7_mapping_re, uni_emotion_masking)
    data_2_multi_label = multi_label_pred
    data_4_34_1_ox = ox
    data_9_34_1_pred = pred
    data_10_34_1_p = p
    data_15_34_7_1_ox = ox_7
    data_16_34_7_1_pred = pred_7
    
    ox, _, pred, p, ox_7, pred_7 = ktop_result(output_34, label, 3, emotion_mapping_34_re, uni_label, label_7_mapping_re, uni_emotion_masking)
    data_5_34_3_ox = ox
    data_11_34_3_pred = pred
    data_12_34_3_p = p
    data_17_34_7_3_ox = ox_7
    data_18_34_7_3_pred = pred_7
    
    ox, _, pred, p, ox_7, pred_7 = ktop_result(output_34, label, 5, emotion_mapping_34_re, uni_label, label_7_mapping_re, uni_emotion_masking)
    data_6_34_5_ox = ox
    data_13_34_5_pred = pred
    data_14_34_5_p = p
    data_19_34_7_5_ox = ox_7
    data_20_34_7_5_pred = pred_7
    
    data_list = [data_1_uni_label, 
                 data_2_multi_label, 
                 data_3_uni_ox, 
                 data_4_34_1_ox, 
                 data_5_34_3_ox, 
                 data_6_34_5_ox, 
                 data_7_uni_pred, 
                 data_8_uni_7_p, 
                 data_9_34_1_pred, 
                 data_10_34_1_p, 
                 data_11_34_3_pred, 
                 data_12_34_3_p, 
                 data_13_34_5_pred, 
                 data_14_34_5_p, 
                 data_15_34_7_1_ox, 
                 data_16_34_7_1_pred, 
                 data_17_34_7_3_ox, 
                 data_18_34_7_3_pred, 
                 data_19_34_7_5_ox, 
                 data_20_34_7_5_pred]
    
    data[name] = data_list 

###################################################################################################

columns = ['7 emotion label', '34 multi emotion label', '7 ox', 'top 1 ox', 'top 3 ox', 'top 5 ox', 
           '7 emotion', '7 prob', 'top 1 emotion', 'top 1 prob', 'top 3 emotion', 'top 3 prob', 'top 5 emotion', 'top 5 prob', 
           'top 1 ox (mapping 7)', 'top 1 emotion (mapping 7)', 'top 3 ox (mapping 7)', 'top 3 emotion (mapping 7)', 'top 5 ox (mapping 7)', 'top 5 emotion (mapping 7)']

df = pd.DataFrame.from_dict(data = data, orient = 'index', columns = columns)
df.index.name = 'audio'
df.to_csv('./result/uni_' + model_name_7 + '_multi_' + model_name_34 + '_result.csv', encoding = 'cp949')

###################################################################################################








###################################################################################################






###################################################################################################























































