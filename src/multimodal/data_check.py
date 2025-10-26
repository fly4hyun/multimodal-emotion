











import os
import pandas as pd




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

label_7_mapping = {
    "ang": 0,   # 분노
    "dis": 1,   # 싫음
    "fea": 2,   # 두려움
    "hap": 3,   # 행복
    "sad": 4,   # 슬픔
    "sur": 5,   # 놀람
    "neu": 6,   # 중립
}

uni_emotion_masking = [0, 0, 
                       1, 1, 1, 1, 
                       2, 2, 2, 2, 
                       3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 
                       4, 4, 4, 4, 4, 4, 4, 4, 
                       5, 
                       6, 6, 6]


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



data_path = "./data"
train_csv = pd.read_csv(os.path.join(data_path, 'train_labels.csv'), encoding = 'UTF-8').to_dict(orient = 'list')

train_data_names = train_csv['audio']
train_data_uni_labels = train_csv['label']
train_data_indexes = [i for i in range(len(train_data_names))]

train_data_labels = []
for train_data in train_data_names:
    train_data_labels.append(multi_datas[train_data])
    
train_uni_labels = []

for data_index in train_data_indexes:
    data_name = train_data_names[data_index]

    train_uni_labels.append(label_7_mapping[train_data_uni_labels[data_index]])


data_path = "./data"
test_csv = pd.read_csv(os.path.join(data_path, 'test_labels.csv'), encoding = 'UTF-8').to_dict(orient = 'list')

test_data_names = test_csv['audio']
test_data_uni_labels = test_csv['label']
test_data_indexes = [i for i in range(len(test_data_names))]

test_data_labels = []
for test_data in test_data_names:
    test_data_labels.append(multi_datas[test_data])

test_uni_labels = []

for data_index in test_data_indexes:
    data_name = test_data_names[data_index]
    test_uni_labels.append(label_7_mapping[test_data_uni_labels[data_index]])

train_data_zip = zip(train_data_labels, train_uni_labels)
test_data_zip = zip(test_data_labels, test_uni_labels)

train_total = 0
train_correct = 0
test_total = 0
test_correct = 0

for data_one in train_data_zip:
    multi, uni = data_one
    
    temp = 0
    for multi_each in multi:
        if uni_emotion_masking[multi_each] == uni:
            temp = temp + 1
    if temp > 0.1:
        train_correct = train_correct + 1
    train_total = train_total + 1

for data_one in test_data_zip:
    multi, uni = data_one
    
    temp = 0
    for multi_each in multi:
        if uni_emotion_masking[multi_each] == uni:
            temp = temp + 1
            print(multi_each, uni)
    if temp > 0.1:
        print(multi, uni)
        test_correct = test_correct + 1
    test_total = test_total + 1
    
total = train_total + test_total
correct = train_correct + test_correct

print('train : {:1.5f}% ( {:5d} / {:5d} )'.format(train_correct / train_total * 100, train_correct, train_total))
print('test  : {:1.5f}% ( {:5d} / {:5d} )'.format(test_correct / test_total * 100, test_correct, test_total))
print('all   : {:1.5f}% ( {:5d} / {:5d} )'.format(correct / total * 100, correct, total))













