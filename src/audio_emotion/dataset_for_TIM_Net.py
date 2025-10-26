







import numpy as np
import pandas as pd
import os
import librosa
from tqdm import tqdm

def get_feature(file_path: str, mfcc_len: int = 39, mean_signal_length: int = 480000):
    """
    file_path: Speech signal folder
    mfcc_len: MFCC coefficient length
    mean_signal_length: MFCC feature average length
    """
    signal, fs = librosa.load(file_path)
    s_len = len(signal)

    if s_len < mean_signal_length:
        pad_len = mean_signal_length - s_len
        pad_rem = pad_len % 2
        pad_len //= 2
        signal = np.pad(signal, (pad_len, pad_len + pad_rem), 'constant', constant_values = 0)
    else:
        pad_len = s_len - mean_signal_length
        pad_len //= 2
        signal = signal[pad_len:pad_len + mean_signal_length]
    mfcc = librosa.feature.mfcc(y=signal, sr=fs, n_mfcc=39)
    mfcc = mfcc.T
    feature = mfcc
    return feature

## 7종 감정
# label_7_mapping = {
#     "ang": 0,   # 분노
#     "dis": 1,   # 싫음
#     "fea": 2,   # 두려움
#     "hap": 3,   # 행복
#     "sad": 4,   # 슬픔
#     "sur": 5,   # 놀람
#     "neu": 6,   # 중립
# }

label_7_mapping = {
    "hap": 0,   # 분노
    "sad": 1,   # 싫음
    "ang": 2,   # 두려움
    "sur": 3,   # 행복
    "neu": 4,   # 슬픔
    "dis": 5,   # 놀람
    "fea": 6,   # 중립
}

one_hot_mapping = {
    0: [1., 0., 0., 0., 0., 0., 0.], 
    1: [0., 1., 0., 0., 0., 0., 0.], 
    2: [0., 0., 1., 0., 0., 0., 0.], 
    3: [0., 0., 0., 1., 0., 0., 0.], 
    4: [0., 0., 0., 0., 1., 0., 0.], 
    5: [0., 0., 0., 0., 0., 1., 0.], 
    6: [0., 0., 0., 0., 0., 0., 1.], 
}

data_path = 'studio'

data_info = pd.read_csv(os.path.join(data_path, "labels.csv"), encoding = "CP949").to_dict(orient = 'list')
#UTF-8
#CP949
data_names_one = data_info["audio"]
labels_one = data_info["label"]

data_dict = {}

iterations = tqdm(range(len(data_names_one)))
for i in iterations:
    
    path = os.path.join(data_path, 'audio', data_names_one[i])
    
    feature = get_feature(path)
    label = one_hot_mapping[label_7_mapping[labels_one[i]]]
    label = np.array(label, dtype=np.float32)
    
    if i == 0:
        datas = np.expand_dims(feature, axis = 0)
        labels = np.expand_dims(label, axis = 0)
    else:
        datas = np.concatenate((datas, np.expand_dims(feature, axis = 0)), axis = 0)
        labels = np.concatenate((labels, np.expand_dims(label, axis = 0)), axis = 0)

        
data_dict['x'] = datas
data_dict['y'] = labels

np_path = './TIM-Net_SER/Code/MFCC/' + data_path + '.npy'
np.save(np_path, data_dict)














