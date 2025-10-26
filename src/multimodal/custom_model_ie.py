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

class Multi_Modal_TransformerEncoder2(nn.Module):
    def __init__(self, num_classes, device):
        super(Multi_Modal_TransformerEncoder2, self).__init__()
        
        self.ie_fc = nn.Linear(1536, 1000)
        
        self.encoder = nn.TransformerEncoderLayer(1000, 8)
        self.transformerencoder = nn.TransformerEncoder(self.encoder, num_layers=1)

        self.classifier = nn.Sequential(
                                        nn.Linear(in_features=1000, out_features=100, bias=True), 
                                        nn.GELU(), 
                                        nn.Dropout1d(0.1), 
                                        nn.Linear(in_features=100, out_features=num_classes, bias=True)
        )

    def forward(self, ief):
        
        ief = self.ie_fc(ief)

        ief = ief.unsqueeze(1)

        ief = torch.cat([ief for _ in range(8)], dim = 1)
        
        #o = torch.cat([ief, sef, tef], dim = 2)
        
        o = self.transformerencoder(ief)[:, 0, :]

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
                                        nn.Linear(in_features=100, out_features=num_classes, bias=True)
        )

    def forward(self, ief):

        ief = ief.squeeze(1)
        
        ief = self.ie_fc(ief)
        
        o = ief * ief
        
        o = self.drop(o)
        o = self.sumpool(o)

        o = torch.nn.functional.normalize(o, p = 0.5)

        o = torch.nn.functional.normalize(o, p = 2)

        return self.classifier(o)

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