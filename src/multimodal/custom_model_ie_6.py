###################################################################################################

import torch
from torch import nn
import torch.nn.functional as F

###################################################################################################

class Multi_Modal_TransformerEncoder(nn.Module):
    def __init__(self, num_classes, device):
        super(Multi_Modal_TransformerEncoder, self).__init__()
        
        self.encoder = nn.TransformerEncoderLayer(1024 + 768, 1)
        self.transformerencoder = nn.TransformerEncoder(self.encoder, num_layers=1)

        self.classifier = nn.Sequential(
                                        nn.Linear(in_features=1024 + 768, out_features=100, bias=True), 
                                        nn.GELU(), 
                                        nn.Dropout1d(0.1), 
                                        nn.Linear(in_features=100, out_features=num_classes, bias=True)
        )

    def forward(self, sef, tef):
        
        sef = sef.squeeze(1)
        tef = tef.squeeze(1)
        
        o = torch.cat([sef, tef], dim = 1)
        
        o = self.transformerencoder(o)
        
        o = o.squeeze(1)

        return self.classifier(o)
    
###################################################################################################

class Multi_Modal_TransformerEncoder3(nn.Module):
    def __init__(self, num_classes, device):
        super(Multi_Modal_TransformerEncoder3, self).__init__()
        
        self.encoder = nn.TransformerEncoderLayer(1536 + 1024, 8)
        self.transformerencoder = nn.TransformerEncoder(self.encoder, num_layers=2)

        self.classifier = nn.Sequential(
                                        nn.Linear(in_features=1536 + 1024, out_features=100, bias=True), 
                                        nn.GELU(), 
                                        nn.Dropout1d(0.1), 
                                        nn.Linear(in_features=100, out_features=num_classes, bias=True)
        )

    def forward(self, ief, sef):
        
        ief = ief.squeeze(1)
        sef = sef.squeeze(1)
        
        o = torch.cat([ief, sef], dim = 1)
        
        o = self.transformerencoder(o)
        
        o = o.squeeze(1)

        return self.classifier(o)

###################################################################################################

class Multi_Modal_MFB(nn.Module):
    def __init__(self, num_classes, device):
        super(Multi_Modal_MFB, self).__init__()
        
        self.ie_fc = nn.Linear(1536, 5000)
        
        self.drop = nn.Dropout1d(0.1)
        
        self.sumpool = nn.AvgPool1d(5)

        self.classifier = nn.Sequential(
                                        nn.Linear(in_features=1000, out_features=100, bias=True), 
                                        nn.GELU(), 
                                        nn.Dropout1d(0.1), 
                                        nn.Linear(in_features=100, out_features=num_classes - 1, bias=True)
        )
        
        self.threshold = nn.Parameter(torch.zeros(6)).to(device)
        self.fc = nn.Linear(in_features=num_classes - 1, out_features=1, bias=True)

    def forward(self, ief):

        ief = ief.squeeze(1)
        
        ief = self.ie_fc(ief)
        
        o = ief * ief
        
        o = self.drop(o)
        o = self.sumpool(o)

        o = torch.nn.functional.normalize(o, p = 0.5)

        o = torch.nn.functional.normalize(o, p = 2)
        
        o = self.classifier(o)
        
        #o = F.softmax(o)
        
        #o_neu = (o * self.threshold).sum(-1).unsqueeze(1)
        o_neu = self.fc(o)

        o = torch.cat([o, o_neu], dim = 1)
        
        #o = F.softmax(o)

        return o, self.threshold

###################################################################################################

class Multi_Modal_MFB1(nn.Module):
    def __init__(self, num_classes, device):
        super(Multi_Modal_MFB1, self).__init__()
        
        self.ie_fc = nn.Linear(1536, 5000)
        
        self.drop = nn.Dropout1d(0.1)
        
        self.sumpool = nn.AvgPool1d(5)

        self.classifier = nn.Sequential(
                                        nn.Linear(in_features=1000, out_features=100, bias=True), 
                                        nn.GELU(), 
                                        nn.Dropout1d(0.1), 
                                        nn.Linear(in_features=100, out_features=num_classes - 1, bias=True)
        )
        
        self.fc = nn.Sequential(
                                        nn.Linear(in_features=1000, out_features=100, bias=True), 
                                        nn.GELU(), 
                                        nn.Dropout1d(0.1), 
                                        nn.Linear(in_features=100, out_features=2, bias=True)
        )
        
        self.threshold = nn.Parameter(torch.zeros(6)).to(device)
        #self.fc = nn.Linear(in_features=num_classes - 1, out_features=1, bias=True)

    def forward(self, ief):

        ief = ief.squeeze(1)
        
        ief = self.ie_fc(ief)
        
        o = ief * ief
        
        o = self.drop(o)
        o = self.sumpool(o)

        o = torch.nn.functional.normalize(o, p = 0.5)

        o = torch.nn.functional.normalize(o, p = 2)
        
        o_6 = self.classifier(o)
        
        #o = F.softmax(o)
        
        #o_neu = (o * self.threshold).sum(-1).unsqueeze(1)
        o_neu = self.fc(o)
        o_neu = F.softmax(o_neu)
        #print(o_neu.shape)
        o = torch.cat([o_6, o_neu[:, 1].unsqueeze(1)], dim = 1)
        
        #o = F.softmax(o)

        return o#, o_neu

###################################################################################################

class Multi_Modal_MFB1(nn.Module):
    def __init__(self, num_classes, device):
        super(Multi_Modal_MFB1, self).__init__()
        
        self.ie_fc = nn.Linear(1536, 5000)
        
        self.drop = nn.Dropout1d(0.1)
        
        self.sumpool = nn.AvgPool1d(5)

        self.classifier = nn.Sequential(
                                        nn.Linear(in_features=1000, out_features=100, bias=True), 
                                        nn.GELU(), 
                                        nn.Dropout1d(0.1), 
                                        nn.Linear(in_features=100, out_features=num_classes, bias=True)
        )
        
        self.fc1 = nn.Sequential(
                                        nn.Linear(in_features=1000, out_features=100, bias=True), 
                                        nn.GELU(), 
                                        nn.Dropout1d(0.1), 
                                        nn.Linear(in_features=100, out_features=2, bias=True)
        )
        
        self.fc2 = nn.Sequential(
                                        nn.Linear(in_features=1000, out_features=100, bias=True), 
                                        nn.GELU(), 
                                        nn.Dropout1d(0.1), 
                                        nn.Linear(in_features=100, out_features=2, bias=True)
        )
        
        self.fc3 = nn.Sequential(
                                        nn.Linear(in_features=1000, out_features=100, bias=True), 
                                        nn.GELU(), 
                                        nn.Dropout1d(0.1), 
                                        nn.Linear(in_features=100, out_features=2, bias=True)
        )
        
        self.fc4 = nn.Sequential(
                                        nn.Linear(in_features=1000, out_features=100, bias=True), 
                                        nn.GELU(), 
                                        nn.Dropout1d(0.1), 
                                        nn.Linear(in_features=100, out_features=2, bias=True)
        )
        
        self.fc5 = nn.Sequential(
                                        nn.Linear(in_features=1000, out_features=100, bias=True), 
                                        nn.GELU(), 
                                        nn.Dropout1d(0.1), 
                                        nn.Linear(in_features=100, out_features=2, bias=True)
        )
        
        self.fc6 = nn.Sequential(
                                        nn.Linear(in_features=1000, out_features=100, bias=True), 
                                        nn.GELU(), 
                                        nn.Dropout1d(0.1), 
                                        nn.Linear(in_features=100, out_features=2, bias=True)
        )
        
        self.fc7 = nn.Sequential(
                                        nn.Linear(in_features=1000, out_features=100, bias=True), 
                                        nn.GELU(), 
                                        nn.Dropout1d(0.1), 
                                        nn.Linear(in_features=100, out_features=2, bias=True)
        )
        
        self.threshold = nn.Parameter(torch.zeros(6)).to(device)
        #self.fc = nn.Linear(in_features=num_classes - 1, out_features=1, bias=True)

    def forward(self, ief):

        ief = ief.squeeze(1)
        
        ief = self.ie_fc(ief)
        
        o = ief * ief
        
        o = self.drop(o)
        o = self.sumpool(o)

        o = torch.nn.functional.normalize(o, p = 0.5)

        o = torch.nn.functional.normalize(o, p = 2)
        
        o_7 = self.classifier(o)
        
        #o = F.softmax(o)
        
        #o_neu = (o * self.threshold).sum(-1).unsqueeze(1)
        #o_neu = self.fc(o)
        o1 = self.fc1(o)
        o2 = self.fc2(o)
        o3 = self.fc3(o)
        o4 = self.fc4(o)
        o5 = self.fc5(o)
        o6 = self.fc6(o)
        o7 = self.fc7(o)

        o1 = F.softmax(o1)
        o2 = F.softmax(o2)
        o3 = F.softmax(o3)
        o4 = F.softmax(o4)
        o5 = F.softmax(o5)
        o6 = F.softmax(o6)
        o7 = F.softmax(o7)
        #print(o_neu.shape)
        o = torch.cat([o1[:, 1].unsqueeze(1), 
                       o2[:, 1].unsqueeze(1), 
                       o3[:, 1].unsqueeze(1), 
                       o4[:, 1].unsqueeze(1), 
                       o5[:, 1].unsqueeze(1), 
                       o6[:, 1].unsqueeze(1), 
                       o7[:, 1].unsqueeze(1)], dim = 1)
        o = o# + o_7
        #o = F.softmax(o)

        return o#, o1, o2, o3, o4, o5, o6, o7
    
###################################################################################################

class Multi_Modal_MFB_transformer(nn.Module):
    def __init__(self, num_classes, device):
        super(Multi_Modal_MFB_transformer, self).__init__()
        
        self.ie_fc = nn.Linear(1536, 5000)
        
        self.drop = nn.Dropout1d(0.1)
        self.sumpool = nn.AvgPool1d(5)
        
        self.encoder_i = nn.TransformerEncoderLayer(1000 + 1536, 1)
        self.transformerencoder_i = nn.TransformerEncoder(self.encoder_i, num_layers=1)

        self.classifier = nn.Sequential(
                                        nn.Linear(in_features=1000 + 1536, out_features=200, bias=True), 
                                        nn.GELU(), 
                                        nn.Dropout1d(0.1), 
                                        nn.Linear(in_features=200, out_features=num_classes, bias=True)
        )

    def forward(self, ief):
        
        ief = ief.squeeze(1)
        
        o_i = self.ie_fc(ief)
        
        o_i = o_i * o_i
        
        o_i = self.sumpool(self.drop(o_i))
        o_i = torch.nn.functional.normalize(o_i, p = 0.5)
        o_i = torch.nn.functional.normalize(o_i, p = 2)

        o_i = torch.cat([o_i, ief], dim = -1)
        
        o_i = self.transformerencoder_i(o_i)

        return self.classifier(o_i)

###################################################################################################

class Multi_Modal_custom(nn.Module):
    def __init__(self, num_classes, device):
        super(Multi_Modal_custom, self).__init__()
        
        self.ie_fc = nn.Linear(1536, 5000)
        
        self.drop = nn.Dropout1d(0.1)
        self.sumpool = nn.AvgPool1d(5)
        
        self.encoder_i = nn.TransformerEncoderLayer(1000 + 1536, 1)
        self.transformerencoder_i = nn.TransformerEncoder(self.encoder_i, num_layers=1)

        self.classifier = nn.Sequential(
                                        nn.Linear(in_features=1000 + 1536, out_features=200, bias=True), 
                                        nn.GELU(), 
                                        nn.Dropout1d(0.1), 
                                        nn.Linear(in_features=200, out_features=num_classes, bias=True)
        )

    def forward(self, ief):
        
        ief = ief.squeeze(1)
        
        o_i = self.ie_fc(ief)
        
        o_i = o_i * o_i
        
        o_i = self.sumpool(self.drop(o_i))
        o_i = torch.nn.functional.normalize(o_i, p = 0.5)
        o_i = torch.nn.functional.normalize(o_i, p = 2)

        o_i = torch.cat([o_i, ief], dim = -1)
        
        o_i = self.transformerencoder_i(o_i)

        return self.classifier(o_i)
    
###################################################################################################