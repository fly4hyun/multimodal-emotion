###################################################################################################

import warnings
warnings.filterwarnings(action='ignore')

from tqdm import tqdm
import numpy as np

import torch
from torch.utils.data import DataLoader
from torch import nn, optim

import dataloader_34
import custom_model_ie
from utils import (MeanAcc, Mean7Acc, Result_MeanACC)

###################################################################################################

k_top = 5

num_classes = 34
num_uniemotion = 7
batch_size = 128
epochs = 200

lr = 0.0001

###################################################################################################

device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(234)
if device == 'cuda:1':
    torch.cuda.manual_seed_all(234)
print(device + ' is avaulable')

###################################################################################################

train_loader = DataLoader(dataloader_34.trainset, batch_size = batch_size, shuffle = True, num_workers = 3, pin_memory = True)
valid_loader = DataLoader(dataloader_34.validset, batch_size = batch_size, shuffle = False, num_workers = 3, pin_memory = True)
test_loader = DataLoader(dataloader_34.testset, batch_size = batch_size, shuffle = False, num_workers = 3, pin_memory = True)

###################################################################################################

model_name = "Multi_Modal_MFB"
model = custom_model_ie.Multi_Modal_MFB(num_classes, device)
model = model.to(device)

###################################################################################################

emotion_mapping_34 = {
    
    'ang': 0, 'unf': 1,

    'dis': 2, 'dul': 3, 'hat': 4, 'irr': 5,

    'fea': 6, 'ash': 7, 'anx': 8, 'emb': 9,

    'hap': 10, 'mis': 11, 'ple': 12, 'lov': 13, 'mag': 14, 'mov': 15,
    'joy': 16, 'lit': 17, 'rom': 18, 'stf': 19, 'cfd': 20, 'thx': 21,

    'sad': 22, 'sor': 23, 'reg': 24, 'lon': 25, 'dip': 26,
    'pai': 27, 'env': 28, 'err': 29,

    'sur': 30,

    'neu': 31, 'wis': 32, 'ind': 33
}

# emotion_mapping_34 = {
    
#     0: 'ang', 1: 'unf',

#     2: 'dis', 3: 'dul', 4: 'hat', 5: 'irr',

#     6: 'fea', 7: 'ash', 8: 'anx', 9: 'emb',

#     10: 'hap', 11: 'mis', 12: 'ple', 13: 'lov', 14: 'mag', 15: 'mov',
#     16: 'joy', 17: 'lit', 18: 'rom', 19: 'stf', 20: 'cfd', 21: 'thx',

#     22: 'sad', 23: 'sor', 24: 'reg', 25: 'lon', 26: 'dip',
#     27: 'pai', 28: 'env', 29: 'reg',

#     30: 'sur',

#     31: 'neu', 32: 'wis', 33: 'ind'
# }

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

uni_emotion_masking = [0, 0, 
                       1, 1, 1, 1, 
                       2, 2, 2, 2, 
                       3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 
                       4, 4, 4, 4, 4, 4, 4, 4, 
                       5, 
                       6, 6, 6]
uni_emotion_masking = torch.LongTensor(uni_emotion_masking).to(device)

###################################################################################################

sample_num = [2739, 438, 9895, 402, 667, 7391, 871, 17, 1766, 3010, 
              1781, 47, 2269, 92, 104, 139, 2153, 324, 351, 1975, 
              548, 232, 1970, 313, 2373, 80, 931, 969, 21, 109, 
              2415, 6492, 173, 133]
sample_weight = [1 - (x / sum(sample_num)) for x in sample_num]
sample_weight = torch.FloatTensor(sample_weight).to(device)

optimizer = optim.Adam(model.parameters(), lr = lr, weight_decay = 1e-8, betas = [0.93, 0.98])
#criterion = nn.MultiLabelSoftMarginLoss(weight = sample_weight, label_smoothing = 0.1)
criterion = nn.MultiLabelSoftMarginLoss(weight = sample_weight)
scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

###################################################################################################

print('Training Start')

best_valid_loss = 99999999
best_epoch = 0
best_tests = 0
best_test_loss = 0
best_test_correct = 0
best_test_total = 1

for epoch in range(epochs):
    
    train_loss_list = []
    valid_loss_list = []
    test_loss_list = []
    
    train_loss = 0
    train_correct = 0
    train_total = 0
    
    valid_loss = 0
    valid_correct = 0
    valid_total = 0
    
    test_loss = 0
    test_correct = 0
    test_total = 0
    
    trains_correct = [0 for i in range(num_classes)]
    valids_correct = [0 for i in range(num_classes)]
    tests_correct = [0 for i in range(num_classes)]
    
    trains_total = [0 for i in range(num_classes)]
    valids_total = [0 for i in range(num_classes)]
    tests_total = [0 for i in range(num_classes)]
    
    trains_reverse_correct = [0 for i in range(num_classes)]
    valids_reverse_correct = [0 for i in range(num_classes)]
    tests_reverse_correct = [0 for i in range(num_classes)]
    
    trains_reverse_total = [0 for i in range(num_classes)]
    valids_reverse_total = [0 for i in range(num_classes)]
    tests_reverse_total = [0 for i in range(num_classes)]

    train_uni_correct = 0
    train_uni_total = 0
    
    valid_uni_correct = 0
    valid_uni_total = 0
    
    test_uni_correct = 0
    test_uni_total = 0

    trains_uni_correct = [0 for i in range(num_uniemotion)]
    valids_uni_correct = [0 for i in range(num_uniemotion)]
    tests_uni_correct = [0 for i in range(num_uniemotion)]
    
    trains_uni_total = [0 for i in range(num_uniemotion)]
    valids_uni_total = [0 for i in range(num_uniemotion)]
    tests_uni_total = [0 for i in range(num_uniemotion)]
    
    trains_reverse_uni_correct = [0 for i in range(num_uniemotion)]
    valids_reverse_uni_correct = [0 for i in range(num_uniemotion)]
    tests_reverse_uni_correct = [0 for i in range(num_uniemotion)]
    
    trains_reverse_uni_total = [0 for i in range(num_uniemotion)]
    valids_reverse_uni_total = [0 for i in range(num_uniemotion)]
    tests_reverse_uni_total = [0 for i in range(num_uniemotion)]
    
    model.train()
    
    iterations = tqdm(train_loader)
    for train_data in iterations:
        
        train_ie_feature = train_data['ie'].to(device)
        train_label = train_data['label'].type(torch.LongTensor).to(device)
        train_uni_label = train_data['uni_label'].type(torch.LongTensor).to(device)

        optimizer.zero_grad()
        
        train_output = model(train_ie_feature)
        loss = criterion(train_output, train_label)

        loss.backward()
        optimizer.step()
        
        train_loss_list.append(loss.item())
        
        with torch.no_grad():
            train_correct = train_correct + (((train_output >= train_output.sort(-1)[0][:, -k_top].unsqueeze(1)) * train_label).sum(-1) > 0.1).sum()
            train_total = train_total + len(train_label)
            
            trains_correct, trains_total, trains_reverse_correct, trains_reverse_total = MeanAcc(trains_correct, trains_total, trains_reverse_correct, trains_reverse_total, train_output, train_label, num_classes, k_top)

            uni_temp_list = []
            for i in range(num_uniemotion):
                
                uni_temp = (train_output >= train_output.sort(-1)[0][:, -k_top].unsqueeze(1))
                uni_temp_list.append(((uni_temp * (uni_emotion_masking == i)).sum(-1) > 0.1).unsqueeze(1) * 1.0)

            train_uni_output = torch.cat(uni_temp_list, dim = 1)
            
            train_uni_correct = train_uni_correct + ((train_uni_output * train_uni_label).sum(-1) >= 0.1).sum()
            train_uni_total = train_uni_total + len(train_uni_label)
            
            trains_uni_correct, trains_uni_total, trains_reverse_uni_correct, trains_reverse_uni_total = Mean7Acc(trains_uni_correct, trains_uni_total, trains_reverse_uni_correct, trains_reverse_uni_total, train_uni_output, train_uni_label, num_uniemotion, k_top)

        pbar_desc = "Model train loss --- "
        pbar_desc += f"Total loss: {np.mean(train_loss_list):.5f}"
        pbar_desc += f" --- Total acc: {(train_correct / train_total):.5f}"
        iterations.set_description(pbar_desc)

    train_loss = np.mean(train_loss_list)
    model.eval()
    scheduler.step()
    
    with torch.no_grad():
        iterations = tqdm(valid_loader)
        for valid_data in iterations:
            
            valid_ie_feature = valid_data['ie'].to(device)
            valid_label = valid_data['label'].type(torch.LongTensor).to(device)
            valid_uni_label = valid_data['uni_label'].type(torch.LongTensor).to(device)
            
            valid_output = model(valid_ie_feature)
            
            loss = criterion(valid_output, valid_label)
            
            valid_loss_list.append(loss.item())
            
            valid_correct = valid_correct + (((valid_output >= valid_output.sort(-1)[0][:, -k_top].unsqueeze(1)) * valid_label).sum(-1) > 0.1).sum()
            valid_total = valid_total + len(valid_label)
            
            valids_correct, valids_total, valids_reverse_correct, valids_reverse_total = MeanAcc(valids_correct, valids_total, valids_reverse_correct, valids_reverse_total, valid_output, valid_label, num_classes, k_top)
                        
            uni_temp_list = []
            for i in range(num_uniemotion):
                
                uni_temp = (valid_output >= valid_output.sort(-1)[0][:, -k_top].unsqueeze(1))
                uni_temp_list.append(((uni_temp * (uni_emotion_masking == i)).sum(-1) > 0.1).unsqueeze(1) * 1.0)

            valid_uni_output = torch.cat(uni_temp_list, dim = 1)
            
            valid_uni_correct = valid_uni_correct + ((valid_uni_output * valid_uni_label).sum(-1) >= 0.1).sum()
            valid_uni_total = valid_uni_total + len(valid_uni_label)
            
            valids_uni_correct, valids_uni_total, valids_reverse_uni_correct, valids_reverse_uni_total = Mean7Acc(valids_uni_correct, valids_uni_total, valids_reverse_uni_correct, valids_reverse_uni_total, valid_uni_output, valid_uni_label, num_uniemotion, k_top)

            pbar_desc = "Model valid loss --- "
            pbar_desc += f"Total loss: {np.mean(valid_loss_list):.5f}"
            pbar_desc += f" --- Total acc: {(valid_correct / valid_total):.5f}"
            iterations.set_description(pbar_desc)
            
        valid_loss = np.mean(valid_loss_list)
    
    with torch.no_grad():
        iterations = tqdm(test_loader)
        for test_data in iterations:
            
            test_ie_feature = test_data['ie'].to(device)
            test_label = test_data['label'].type(torch.LongTensor).to(device)
            test_uni_label = test_data['uni_label'].type(torch.LongTensor).to(device)
            
            test_output = model(test_ie_feature)
            
            loss = criterion(test_output, test_label)
            
            test_loss_list.append(loss.item())
            
            test_correct = test_correct + (((test_output >= test_output.sort(-1)[0][:, -k_top].unsqueeze(1)) * test_label).sum(-1) > 0.1).sum()
            test_total = test_total + len(test_label)
            
            tests_correct, tests_total, tests_reverse_correct, tests_reverse_total = MeanAcc(tests_correct, tests_total, tests_reverse_correct, tests_reverse_total, test_output, test_label, num_classes, k_top)
                    
            uni_temp_list = []
            for i in range(num_uniemotion):
                
                uni_temp = (test_output >= test_output.sort(-1)[0][:, -k_top].unsqueeze(1))
                uni_temp_list.append(((uni_temp * (uni_emotion_masking == i)).sum(-1) > 0.1).unsqueeze(1) * 1.0)

            test_uni_output = torch.cat(uni_temp_list, dim = 1)
            
            test_uni_correct = test_uni_correct + ((test_uni_output * test_uni_label).sum(-1) >= 0.1).sum()
            test_uni_total = test_uni_total + len(test_uni_label)
            
            tests_uni_correct, tests_uni_total, tests_reverse_uni_correct, tests_reverse_uni_total = Mean7Acc(tests_uni_correct, tests_uni_total, tests_reverse_uni_correct, tests_reverse_uni_total, test_uni_output, test_uni_label, num_uniemotion, k_top)
            
            pbar_desc = "Model test  loss --- "
            pbar_desc += f"Total loss: {np.mean(test_loss_list):.5f}"
            pbar_desc += f" --- Total acc: {(test_correct / test_total):.5f}"
            iterations.set_description(pbar_desc)
            
        test_loss = np.mean(test_loss_list)
    
    Result_MeanACC(epoch, train_correct, train_total, train_loss, train_uni_correct, train_uni_total, 
                   trains_correct, trains_total, trains_reverse_correct, trains_reverse_total, 
                   trains_uni_correct, trains_uni_total, trains_reverse_uni_correct, trains_reverse_uni_total, 
                   emotion_mapping_34, num_classes, 'train')
    Result_MeanACC(epoch, valid_correct, valid_total, valid_loss, valid_uni_correct, valid_uni_total, 
                   valids_correct, valids_total, valids_reverse_correct, valids_reverse_total, 
                   valids_uni_correct, valids_uni_total, valids_reverse_uni_correct, valids_reverse_uni_total, 
                   emotion_mapping_34, num_classes, 'valid')
    Result_MeanACC(epoch, test_correct, test_total, test_loss, test_uni_correct, test_uni_total, 
                   tests_correct, tests_total, tests_reverse_correct, tests_reverse_total, 
                   tests_uni_correct, tests_uni_total, tests_reverse_uni_correct, tests_reverse_uni_total, 
                   emotion_mapping_34, num_classes, 'test')    
    
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        model_path = './models_34_ie/' + model_name + '_top' + str(k_top) + '.pth'
        model = model.cpu()
        torch.save(model.state_dict(), model_path)
        model = model.to(device)
        
        best_epoch = epoch
        best_test_loss = test_loss
        best_test_correct = test_correct
        best_test_total = test_total
        best_test_uni_correct = test_uni_correct
        best_test_uni_total = test_uni_total
        best_tests_correct = tests_correct
        best_tests_total = tests_total
        best_tests_reverse_correct = tests_reverse_correct
        best_tests_reverse_total = tests_reverse_total
        best_tests_uni_correct = tests_uni_correct
        best_tests_uni_total = tests_uni_total
        best_tests_reverse_uni_correct = tests_reverse_uni_correct
        best_tests_reverse_uni_total = tests_reverse_uni_total

    Result_MeanACC(best_epoch, best_test_correct, best_test_total, best_test_loss, best_test_uni_correct, best_test_uni_total, 
                   best_tests_correct, best_tests_total, best_tests_reverse_correct, best_tests_reverse_total, 
                   best_tests_uni_correct, best_tests_uni_total, best_tests_reverse_uni_correct, best_tests_reverse_uni_total, 
                   emotion_mapping_34, num_classes, 'test')
    
###################################################################################################