import warnings
warnings.filterwarnings(action='ignore')

import os
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from torch import nn, optim

import custom_dataset
import custom_model
from utils import (precision, 
                   recall, 
                   F1score, 
                   print_result
)
                   

os.environ["TOKENIZERS_PARALLELISM"] = "false"

##########################################################################################
##########################################################################################

top_k = 1
num_classes = 7
batch_size = 32
epochs = 300

lr = 0.00001

##########################################################################################

device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(234)
if device == 'cuda':
    torch.cuda.manual_seed_all(234)
print(device + ' is avaulable')

##########################################################################################

train_loader = DataLoader(custom_dataset.trainset, batch_size = batch_size, shuffle = True, num_workers = 3, pin_memory = True)
valid_loader = DataLoader(custom_dataset.validset, batch_size = batch_size, shuffle = False, num_workers = 3, pin_memory = True)
test_loader = DataLoader(custom_dataset.testset, batch_size = batch_size, shuffle = False, num_workers = 3, pin_memory = True)

##########################################################################################

# train_data = next(iter(test_loader))
# x_size_1 = train_data['size_1'][0].item()
# x_size_2 = train_data['size_2'][0].item()

#model_name = 'Emformer_mfcc'
model_name = 'GoogleNet_1d'
#model_name = 'TIMNet'
#model_name = 'Transformer_Model'

#model = custom_model.Emformer_mfcc(num_classes, device)
model = custom_model.GoogleNet_1d(num_classes, device)
#model = custom_model.TIMNet(x_size_1, x_size_2, num_classes, device)
#model = custom_model.Transformer_Model(num_classes, device)
model = model.to(device)

#####################################################################################

label_7_mapping = {
    "ang": 0,   # 분노
    "dis": 1,   # 싫음
    "fea": 2,   # 두려움
    "hap": 3,   # 행복
    "sad": 4,   # 슬픔
    "sur": 5,   # 놀람
    "neu": 6,   # 중립
}

#sample_num = [141413, 21817, 72767, 133315, 235722, 20125, 140568]
sample_num = [10000, 10000, 10000, 10000, 10000, 10000, 10000]
sample_weight = [1 - (x / sum(sample_num)) for x in sample_num]
sample_weight = torch.FloatTensor(sample_weight).to(device)

##########################################################################################

optimizer = optim.Adam(model.parameters(), lr = lr, weight_decay = 1e-8, betas = [0.93, 0.98])
#criterion = nn.MSELoss()
criterion = nn.CrossEntropyLoss(weight = sample_weight, label_smoothing = 0.1)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.2)

##########################################################################################

print('Training Start')

best_valid_loss = 9999999

for epoch in range(epochs):
    
    
    
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

        train_audio = train_data['audio'].to(device)
        train_label = train_data['label'].type(torch.LongTensor).to(device)
        
        optimizer.zero_grad()
        
        output = model(train_audio)
        loss = criterion(output, train_label)

        loss.backward()
        optimizer.step()
        
        train_loss = train_loss + loss.item()
        

        preds = torch.max(output.data, 1)[1]
        #target = torch.max(train_label.data, 1)[1]
        target = train_label.data

        for i in range(num_classes):
            for j in range(num_classes):
                trains[i][j] = trains[i][j] + ((target == i) * (preds == j)).sum().item()
            train_correct = train_correct + ((target == i) * (preds == i)).sum().item()
        train_total = train_total + len(train_label)
            
        pbar_desc = "Model train loss --- "
        pbar_desc += f"Total loss: {loss.item():.5f}"
        iterations.set_description(pbar_desc)

    model.eval()
    
    with torch.no_grad():
        iterations = tqdm(valid_loader)
        for valid_data in iterations:
        
            valid_audio = valid_data['audio'].to(device)
            valid_label = valid_data['label'].type(torch.LongTensor).to(device)
            
            valid_output = model(valid_audio)
            
            loss = criterion(valid_output, valid_label)
            
            valid_loss = valid_loss + loss.item()
            
            preds = torch.max(valid_output.data, 1)[1]
            #target = torch.max(valid_label.data, 1)[1]
            target = valid_label.data
            
            for i in range(num_classes):
                for j in range(num_classes):
                    valids[i][j] = valids[i][j] + ((target == i) * (preds == j)).sum().item()
                valid_correct = valid_correct + ((target == i) * (preds == i)).sum().item()
            valid_total = valid_total + len(valid_label)
                        
            pbar_desc = "Model valid loss --- "
            pbar_desc += f"Total loss: {loss.item():.5f}"
            iterations.set_description(pbar_desc)
            
    scheduler.step()
    
    with torch.no_grad():
        iterations = tqdm(test_loader)
        for test_data in iterations:
        
            test_audio = test_data['audio'].to(device)

            test_label = test_data['label'].type(torch.LongTensor).to(device)
            
            test_output = model(test_audio)
            
            loss = criterion(test_output, test_label)
            
            test_loss = test_loss + loss.item()
            
            preds = torch.max(test_output.data, 1)[1]
            #target = torch.max(test_label.data, 1)[1]
            target = test_label.data
            
            for i in range(num_classes):
                for j in range(num_classes):
                    tests[i][j] = tests[i][j] + ((target == i) * (preds == j)).sum().item()
                test_correct = test_correct + ((target == i) * (preds == i)).sum().item()
            test_total = test_total + len(test_label)
                    
            pbar_desc = "Model test  loss --- "
            pbar_desc += f"Total loss: {loss.item():.5f}"
            iterations.set_description(pbar_desc)
            
    print_result('train', epoch, train_correct, train_total, train_loss, num_classes, trains, label_7_mapping)
    print_result('valid', epoch, valid_correct, valid_total, valid_loss, num_classes, valids, label_7_mapping)
    print_result('test', epoch, test_correct, test_total, test_loss, num_classes, tests, label_7_mapping)
    
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        model_path = './models/' + model_name + '.pth'
        model = model.cpu()
        torch.save(model.state_dict(), model_path)
        model = model.to(device)
        
        best_epoch = epoch
        best_tests = tests
        best_test_loss = test_loss
        best_test_correct = test_correct
        best_test_total = test_total
        
    print_result('test', best_epoch, best_test_correct, best_test_total, best_test_loss, num_classes, best_tests, label_7_mapping)
        
        # saved_model = custom_model.Emformer_mfcc(num_classes, device)
        # saved_model.load_state_dict(torch.load(PATH))