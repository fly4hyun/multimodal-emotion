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
        self.se_fc = nn.Linear(1024, 1000)
        
        self.encoder = nn.TransformerEncoderLayer(1000 + 1000, 8)
        self.transformerencoder = nn.TransformerEncoder(self.encoder, num_layers=1)

        self.classifier = nn.Sequential(
                                        nn.Linear(in_features=1000 + 1000, out_features=100, bias=True), 
                                        nn.GELU(), 
                                        nn.Dropout1d(0.1), 
                                        nn.Linear(in_features=100, out_features=num_classes, bias=True)
        )

    def forward(self, ief, sef):
        
        ief = self.ie_fc(ief)
        sef = self.se_fc(sef)

        ief = ief.unsqueeze(1)
        sef = sef.unsqueeze(1)

        ief = torch.cat([ief for _ in range(8)], dim = 1)
        sef = torch.cat([sef for _ in range(8)], dim = 1)
        
        o = torch.cat([ief, sef], dim = 2)
        
        o = self.transformerencoder(o)[:, 0, :]

        o = o.squeeze(1)

        return self.classifier(o)
    
###################################################################################################

class Multi_Modal_TransformerEncoder4(nn.Module):
    def __init__(self, num_classes, device):
        super(Multi_Modal_TransformerEncoder4, self).__init__()
        
        self.ie_fc = nn.Linear(1536, 1000)
        self.se_fc = nn.Linear(1024, 1000)
        
        self.encoder_ie = nn.TransformerEncoderLayer(1000, 8)
        self.transformerencoder_ie = nn.TransformerEncoder(self.encoder_ie, num_layers=1)
        
        self.encoder_se = nn.TransformerEncoderLayer(1000, 8)
        self.transformerencoder_se = nn.TransformerEncoder(self.encoder_se, num_layers=1)
        
        self.decoder_ie_se = nn.TransformerDecoderLayer(1000, 8)
        self.transformerencoder_ie_se = nn.TransformerDecoder(self.decoder_ie_se, num_layers=1)
        
        self.decoder_se_ie = nn.TransformerDecoderLayer(1000, 8)
        self.transformerencoder_se_ie = nn.TransformerDecoder(self.decoder_se_ie, num_layers=1)
        
        self.encoder = nn.TransformerEncoderLayer(2000, 8)
        self.transformerencodee = nn.TransformerEncoder(self.encoder, num_layers=1)

        self.classifier = nn.Sequential(
                                        nn.Linear(in_features=1000 + 1000, out_features=100, bias=True), 
                                        nn.GELU(), 
                                        nn.Dropout1d(0.1), 
                                        nn.Linear(in_features=100, out_features=num_classes, bias=True)
        )

    def forward(self, ief, sef):
        
        ief = self.ie_fc(ief)
        sef = self.se_fc(sef)

        ief = ief.unsqueeze(1)
        sef = sef.unsqueeze(1)

        ief = torch.cat([ief for _ in range(8)], dim = 1)
        sef = torch.cat([sef for _ in range(8)], dim = 1)
        
        #iief = self.transformerencoder_ie(ief)[:, 0, :]
        #ssef = self.transformerencoder_se(sef)[:, 0, :]
        isef = self.transformerencoder_ie_se(ief, sef)#[:, 0, :]
        sief = self.transformerencoder_se_ie(sef, ief)#[:, 0, :]
        
        #ief = iief * isef
        #sef = ssef * sief
        o = torch.cat([isef, sief], dim = 2)
        
        o = self.transformerencodee(o)[:, 0, :]
        
        #o = torch.cat([ief, sef], dim = 1)
        
        #o = self.transformerencoder(o)[:, 0, :]

        #o = o.squeeze(1)

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
        self.se_fc = nn.Linear(1024, 5000)
        
        self.drop = nn.Dropout1d(0.1)
        
        self.sumpool = nn.AvgPool1d(5)

        self.classifier = nn.Sequential(
                                        nn.Linear(in_features=1000, out_features=100, bias=True), 
                                        nn.GELU(), 
                                        nn.Dropout1d(0.1), 
                                        nn.Linear(in_features=100, out_features=num_classes, bias=True)
        )

    def forward(self, ief, sef):

        ief = ief.squeeze(1)
        sef = sef.squeeze(1)
        
        ief = self.ie_fc(ief)
        sef = self.se_fc(sef)
        
        o = ief * sef
        
        o = self.drop(o)
        o = self.sumpool(o)

        o = torch.nn.functional.normalize(o, p = 0.5)

        o = torch.nn.functional.normalize(o, p = 2)

        return self.classifier(o)

###################################################################################################

class Multi_Modal_Custom(nn.Module):
    def __init__(self, num_classes, device):
        super(Multi_Modal_Custom, self).__init__()
        
        self.ie_fc = nn.Linear(1536, 5000)
        self.se_fc = nn.Linear(1024, 5000)
        
        self.drop = nn.Dropout1d(0.1)
        
        self.sumpool = nn.AvgPool1d(5)

        self.classifier = nn.Sequential(
                                        nn.Linear(in_features=1000, out_features=100, bias=True), 
                                        nn.GELU(), 
                                        nn.Dropout1d(0.1), 
                                        nn.Linear(in_features=100, out_features=num_classes, bias=True)
        )

    def forward(self, ief, sef):

        ief = ief.squeeze(1)
        sef = sef.squeeze(1)
        
        ief = self.ie_fc(ief)
        sef = self.se_fc(sef)
        
        o = ief * sef
        
        o = self.drop(o)
        o = self.sumpool(o)

        o = torch.nn.functional.normalize(o, p = 0.5)

        o = torch.nn.functional.normalize(o, p = 2)

        return self.classifier(o)

###################################################################################################



