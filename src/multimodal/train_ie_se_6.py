###################################################################################################

import warnings
warnings.filterwarnings(action='ignore')

from tqdm import tqdm
import numpy as np

import torch
from torch.utils.data import DataLoader
from torch import nn, optim

import dataloader_6
import custom_model_ie_se_6
from utils import (print_result)

###################################################################################################

num_classes = 7
batch_size = 128
epochs = 200

lr = 0.0001

###################################################################################################

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(234)
if device == 'cuda:0':
    torch.cuda.manual_seed_all(234)
print(device + ' is avaulable')

###################################################################################################

train_loader = DataLoader(dataloader_6.trainset, batch_size = batch_size, shuffle = True, num_workers = 3, pin_memory = True)
valid_loader = DataLoader(dataloader_6.validset, batch_size = batch_size, shuffle = False, num_workers = 3, pin_memory = True)
test_loader = DataLoader(dataloader_6.testset, batch_size = batch_size, shuffle = False, num_workers = 3, pin_memory = True)

###################################################################################################

model_name = "Multi_Modal_Custom"
model = custom_model_ie_se_6.Multi_Modal_Custom(num_classes, device)
model = model.to(device)

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

#sample_num = [10241, 16202, 6578, 15447, 9534, 11539, 14337]
sample_num = [4767, 4642, 3356, 4167, 4715, 3797, 4770]
sample_weight = [1 - (x / sum(sample_num)) for x in sample_num]
sample_weight = torch.FloatTensor(sample_weight).to(device)

optimizer = optim.Adam(model.parameters(), lr = lr, weight_decay = 1e-8, betas = [0.93, 0.98])
criterion = nn.CrossEntropyLoss(weight = sample_weight, label_smoothing = 0.1)
scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

###################################################################################################

print('Training Start')

best_valid_F1 = 0
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
    
    trains = [[0 for i in range(num_classes)] for j in range(num_classes)]
    valids = [[0 for i in range(num_classes)] for j in range(num_classes)]
    tests = [[0 for i in range(num_classes)] for j in range(num_classes)]
    
    model.train()
    
    iterations = tqdm(train_loader)
    for train_data in iterations:
        
        train_ie_feature = train_data['ie'].to(device)
        train_se_feature = train_data['se'].to(device)
        train_label = train_data['label'].type(torch.LongTensor).to(device)
        
        # optimizer.zero_grad()
        # _, o_i, o_s = model(train_ie_feature, train_se_feature)
        # loss_i = criterion(o_i, train_label)
        # loss_s = criterion(o_s, train_label)
        # loss_f = loss_i + loss_s
        # loss_f.backward()
        # optimizer.step()
        
        # optimizer.zero_grad()
        # train_output, _, _ = model(train_ie_feature, train_se_feature)
        # loss = criterion(train_output, train_label)
        # loss.backward()
        # optimizer.step()
        
        optimizer.zero_grad()
        train_output, o_i, o_s = model(train_ie_feature, train_se_feature)
        loss_i = criterion(o_i, train_label)
        loss_s = criterion(o_s, train_label)
        
        loss = criterion(train_output, train_label)
        
        loss_f = 0.2 * (loss_i + loss_s) + loss
        
        loss.backward()
        optimizer.step()
        
        train_loss_list.append(loss.item())

        preds = torch.max(train_output.data, 1)[1]
        target = train_label.data

        for i in range(num_classes):
            for j in range(num_classes):
                trains[i][j] = trains[i][j] + ((target == i) * (preds == j)).sum().item()
            train_correct = train_correct + ((target == i) * (preds == i)).sum().item()
        train_total = train_total + len(train_label)
            
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
            valid_se_feature = valid_data['se'].to(device)
            valid_label = valid_data['label'].type(torch.LongTensor).to(device)

            
            valid_output, _, _ = model(valid_ie_feature, valid_se_feature)
            
            loss = criterion(valid_output, valid_label)
            
            valid_loss_list.append(loss.item())
            
            preds = torch.max(valid_output.data, 1)[1]
            target = valid_label.data
            
            for i in range(num_classes):
                for j in range(num_classes):
                    valids[i][j] = valids[i][j] + ((target == i) * (preds == j)).sum().item()
                valid_correct = valid_correct + ((target == i) * (preds == i)).sum().item()
            valid_total = valid_total + len(valid_label)
                        
            pbar_desc = "Model valid loss --- "
            pbar_desc += f"Total loss: {np.mean(valid_loss_list):.5f}"
            pbar_desc += f" --- Total acc: {(valid_correct / valid_total):.5f}"
            iterations.set_description(pbar_desc)
            
        valid_loss = np.mean(valid_loss_list)
    
    with torch.no_grad():
        iterations = tqdm(test_loader)
        for test_data in iterations:
            
            test_ie_feature = test_data['ie'].to(device)
            test_se_feature = test_data['se'].to(device)
            test_label = test_data['label'].type(torch.LongTensor).to(device)
            
            test_output, _, _ = model(test_ie_feature, test_se_feature)
            
            loss = criterion(test_output, test_label)
            
            test_loss_list.append(loss.item())
            
            preds = torch.max(test_output.data, 1)[1]
            target = test_label.data
            
            for i in range(num_classes):
                for j in range(num_classes):
                    tests[i][j] = tests[i][j] + ((target == i) * (preds == j)).sum().item()
                test_correct = test_correct + ((target == i) * (preds == i)).sum().item()
            test_total = test_total + len(test_label)
                    
            pbar_desc = "Model test  loss --- "
            pbar_desc += f"Total loss: {np.mean(test_loss_list):.5f}"
            pbar_desc += f" --- Total acc: {(test_correct / test_total):.5f}"
            iterations.set_description(pbar_desc)
            
        test_loss = np.mean(test_loss_list)
            
    _ = print_result('train', epoch, train_correct, train_total, train_loss, num_classes, trains, label_7_mapping)
    valid_F1 = print_result('valid', epoch, valid_correct, valid_total, valid_loss, num_classes, valids, label_7_mapping)
    _ = print_result('test', epoch, test_correct, test_total, test_loss, num_classes, tests, label_7_mapping)
    
    if valid_F1 > best_valid_F1:
        best_valid_F1 = valid_F1
        model_path = './models_ie_se_6/' + model_name + '.pth'
        model = model.cpu()
        torch.save(model.state_dict(), model_path)
        model = model.to(device)
        
        best_epoch = epoch
        best_tests = tests
        best_test_loss = test_loss
        best_test_correct = test_correct
        best_test_total = test_total
        
    _ = print_result('test', best_epoch, best_test_correct, best_test_total, best_test_loss, num_classes, best_tests, label_7_mapping)

###################################################################################################