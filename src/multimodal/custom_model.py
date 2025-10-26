###################################################################################################

import torch
from torch import nn
import torch.nn.functional as F

###################################################################################################

class Multi_Modal_TransformerEncoder(nn.Module):
    def __init__(self, num_classes, device):
        super(Multi_Modal_TransformerEncoder, self).__init__()
        
        self.encoder = nn.TransformerEncoderLayer(1536 + 1024 + 768, 1)
        self.transformerencoder = nn.TransformerEncoder(self.encoder, num_layers=1)

        self.classifier = nn.Sequential(
                                        nn.Linear(in_features=1536 + 1024 + 768, out_features=100, bias=True), 
                                        nn.GELU(), 
                                        nn.Dropout1d(0.1), 
                                        nn.Linear(in_features=100, out_features=num_classes, bias=True)
        )

    def forward(self, ief, sef, tef):
        
        ief = ief.squeeze(1)
        sef = sef.squeeze(1)
        tef = tef.squeeze(1)
        
        o = torch.cat([ief, sef, tef], dim = 1)
        
        o = self.transformerencoder(o)
        
        o = o.squeeze(1)

        return self.classifier(o)
    
###################################################################################################

class Multi_Modal_TransformerEncoder3(nn.Module):
    def __init__(self, num_classes, device):
        super(Multi_Modal_TransformerEncoder3, self).__init__()
        
        self.encoder = nn.TransformerEncoderLayer(1536 + 1024 + 768, 1)
        self.transformerencoder = nn.TransformerEncoder(self.encoder, num_layers=1)
        
        self.encoder_is = nn.TransformerEncoderLayer(1536 + 1024, 1)
        self.transformerencoder_is = nn.TransformerEncoder(self.encoder_is, num_layers=1)
        
        self.encoder_st = nn.TransformerEncoderLayer(1024 + 768, 1)
        self.transformerencoder_st = nn.TransformerEncoder(self.encoder_st, num_layers=1)
        
        self.encoder_ti = nn.TransformerEncoderLayer(768 + 1536, 1)
        self.transformerencoder_ti = nn.TransformerEncoder(self.encoder_ti, num_layers=1)
        

        self.classifier = nn.Sequential(
                                        nn.Linear(in_features=3 * (1536 + 1024 + 768), out_features=100, bias=True), 
                                        nn.GELU(), 
                                        nn.Dropout1d(0.1), 
                                        nn.Linear(in_features=100, out_features=num_classes, bias=True)
        )

    def forward(self, ief, sef, tef):
        
        ief = ief.squeeze(1)
        sef = sef.squeeze(1)
        tef = tef.squeeze(1)
        
        o_ist = torch.cat([ief, sef, tef], dim = 1)
        
        o_is = torch.cat([ief, sef], dim = 1)
        o_st = torch.cat([sef, tef], dim = 1)
        o_ti = torch.cat([tef, ief], dim = 1)
        
        o_ist = self.transformerencoder(o_ist)
        
        o_is = self.transformerencoder_is(o_is)
        o_st = self.transformerencoder_st(o_st)
        o_ti = self.transformerencoder_ti(o_ti)
        
        o = torch.cat([o_ist, o_is, o_st, o_ti], dim = 1)
        
        o = o.squeeze(1)

        return self.classifier(o)

###################################################################################################

class Multi_Modal_TransformerEncoder2(nn.Module):
    def __init__(self, num_classes, device):
        super(Multi_Modal_TransformerEncoder2, self).__init__()
        
        self.ie_fc = nn.Linear(1536, 1000)
        self.se_fc = nn.Linear(1024, 1000)
        self.te_fc = nn.Linear(768, 1000)
        
        self.encoder = nn.TransformerEncoderLayer(1000 + 1000 + 1000, 8)
        self.transformerencoder = nn.TransformerEncoder(self.encoder, num_layers=1)

        self.classifier = nn.Sequential(
                                        nn.Linear(in_features=1000 + 1000 + 1000, out_features=100, bias=True), 
                                        nn.GELU(), 
                                        nn.Dropout1d(0.1), 
                                        nn.Linear(in_features=100, out_features=num_classes, bias=True)
        )

    def forward(self, ief, sef, tef):
        
        ief = self.ie_fc(ief)
        sef = self.se_fc(sef)
        tef = self.te_fc(tef)

        ief = ief.unsqueeze(1)
        sef = sef.unsqueeze(1)
        tef = tef.unsqueeze(1)

        ief = torch.cat([ief for _ in range(8)], dim = 1)
        sef = torch.cat([sef for _ in range(8)], dim = 1)
        tef = torch.cat([tef for _ in range(8)], dim = 1)
        
        o = torch.cat([ief, sef, tef], dim = 2)
        
        o = self.transformerencoder(o)[:, 0, :]

        o = o.squeeze(1)

        return self.classifier(o)
    
###################################################################################################

class Multi_Modal_MFB(nn.Module):
    def __init__(self, num_classes, device):
        super(Multi_Modal_MFB, self).__init__()
        
        self.ie_fc = nn.Linear(1536, 5000)
        self.se_fc = nn.Linear(1024, 5000)
        self.te_fc = nn.Linear(768, 5000)
        
        self.drop = nn.Dropout1d(0.1)
        
        self.sumpool = nn.AvgPool1d(5)

        self.classifier = nn.Sequential(
                                        nn.Linear(in_features=1000, out_features=100, bias=True), 
                                        nn.GELU(), 
                                        nn.Dropout1d(0.1), 
                                        nn.Linear(in_features=100, out_features=num_classes, bias=True)
        )

    def forward(self, ief, sef, tef):
        
        ief = ief.squeeze(1)
        sef = sef.squeeze(1)
        tef = tef.squeeze(1)
        
        ief = self.ie_fc(ief)
        sef = self.se_fc(sef)
        tef = self.te_fc(tef)
        
        o = (ief * sef + sef * tef + tef * ief + ief * sef * tef) / 4
        
        o = self.drop(o)
        o = self.sumpool(o)

        o = torch.nn.functional.normalize(o, p = 0.5)

        o = torch.nn.functional.normalize(o, p = 2)

        return self.classifier(o)

###################################################################################################

class Multi_Modal_MFB1(nn.Module):
    def __init__(self, num_classes, device):
        super(Multi_Modal_MFB1, self).__init__()
        
        self.ie_fc = nn.Linear(1536, 5000)
        self.se_fc = nn.Linear(1024, 5000)
        self.te_fc = nn.Linear(768, 5000)
        
        self.drop = nn.Dropout1d(0.1)
        
        self.sumpool = nn.AvgPool1d(5)

        self.classifier = nn.Sequential(
                                        nn.Linear(in_features=1000, out_features=100, bias=True), 
                                        nn.GELU(), 
                                        nn.Dropout1d(0.1), 
                                        nn.Linear(in_features=100, out_features=num_classes, bias=True)
        )

    def forward(self, ief, sef, tef):
        
        ief = ief.squeeze(1)
        sef = sef.squeeze(1)
        tef = tef.squeeze(1)
        
        ief = self.ie_fc(ief)
        sef = self.se_fc(sef)
        tef = self.te_fc(tef)
        
        o = ief * sef * tef
        
        o = self.drop(o)
        o = self.sumpool(o)

        o = torch.nn.functional.normalize(o, p = 0.5)

        o = torch.nn.functional.normalize(o, p = 2)

        return self.classifier(o)
    
###################################################################################################

class Multi_Modal_MFB2(nn.Module):
    def __init__(self, num_classes, device):
        super(Multi_Modal_MFB2, self).__init__()
        
        self.ie_fc = nn.Linear(1536, 5000)
        self.se_fc = nn.Linear(1024, 5000)
        self.te_fc = nn.Linear(768, 5000)
        
        self.drop = nn.Dropout1d(0.1)
        
        self.sumpool = nn.AvgPool1d(5)

        self.classifier = nn.Sequential(
                                        nn.Linear(in_features=1000, out_features=100, bias=True), 
                                        nn.GELU(), 
                                        nn.Dropout1d(0.1), 
                                        nn.Linear(in_features=100, out_features=num_classes, bias=True)
        )

    def forward(self, ief, sef, tef):
        
        ief = ief.squeeze(1)
        sef = sef.squeeze(1)
        tef = tef.squeeze(1)
        
        ief = self.ie_fc(ief)
        sef = self.se_fc(sef)
        tef = self.te_fc(tef)
        
        o = (ief * sef + sef * tef + tef * ief) / 3
        
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
        self.se_fc = nn.Linear(1024, 5000)
        self.te_fc = nn.Linear(768, 5000)
        
        self.drop = nn.Dropout1d(0.1)
        self.sumpool = nn.AvgPool1d(5)
        
        self.encoder_i = nn.TransformerEncoderLayer(1000 + 1536, 1)
        self.transformerencoder_i = nn.TransformerEncoder(self.encoder_i, num_layers=1)
        
        self.encoder_s = nn.TransformerEncoderLayer(1000 + 1024, 1)
        self.transformerencoder_s = nn.TransformerEncoder(self.encoder_s, num_layers=1)
        
        self.encoder_t = nn.TransformerEncoderLayer(1000 + 768, 1)
        self.transformerencoder_t = nn.TransformerEncoder(self.encoder_t, num_layers=1)
        

        self.classifier = nn.Sequential(
                                        nn.Linear(in_features=1000 * 3 + 1536 + 1024 + 768, out_features=200, bias=True), 
                                        nn.GELU(), 
                                        nn.Dropout1d(0.1), 
                                        nn.Linear(in_features=200, out_features=num_classes, bias=True)
        )

    def forward(self, ief, sef, tef):
        
        ief = ief.squeeze(1)
        sef = sef.squeeze(1)
        tef = tef.squeeze(1)
        
        o_i = self.ie_fc(ief)
        o_s = self.se_fc(sef)
        o_t = self.te_fc(tef)
        
        o_is = o_i * o_s
        o_st = o_s * o_t
        o_ti = o_t * o_i
        
        o_is = self.sumpool(self.drop(o_is))
        o_is = torch.nn.functional.normalize(o_is, p = 0.5)
        o_is = torch.nn.functional.normalize(o_is, p = 2)
        
        o_st = self.sumpool(self.drop(o_st))
        o_st = torch.nn.functional.normalize(o_st, p = 0.5)
        o_st = torch.nn.functional.normalize(o_st, p = 2)
        
        o_ti = self.sumpool(self.drop(o_ti))
        o_ti = torch.nn.functional.normalize(o_ti, p = 0.5)
        o_ti = torch.nn.functional.normalize(o_ti, p = 2)
        
        o_is = torch.cat([o_is, tef], dim = -1)
        o_st = torch.cat([o_st, ief], dim = -1)
        o_ti = torch.cat([o_ti, sef], dim = -1)
        
        o_st = self.transformerencoder_i(o_st)
        o_ti = self.transformerencoder_s(o_ti)
        o_is = self.transformerencoder_t(o_is)
        
        o = torch.cat([o_is, o_st, o_ti], dim = -1)

        return self.classifier(o)
    
###################################################################################################