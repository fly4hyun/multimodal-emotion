###################################################################################################

import math
import numpy as np

import torch
from torch import nn
import torch.nn.functional as F

###################################################################################################

class GraphConvolution(nn.Module):

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

def gen_A(num_classes, t, _adj):
    _adj[_adj < t] = 0
    _adj[_adj >= t] = 1
    _adj = _adj * 0.25 / (_adj.sum(0, keepdims=True) + 1e-6)
    _adj = _adj + np.identity(num_classes, np.int64)
    return _adj

def gen_adj(A):
    D = torch.pow(A.sum(1).float(), -0.5)
    D = torch.diag(D)
    adj = torch.matmul(torch.matmul(A, D).t(), D)
    return adj

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
        
        self.se_fc = nn.Linear(1024, 5000)
        
        self.drop = nn.Dropout1d(0.1)
        
        self.sumpool = nn.AvgPool1d(5)

        self.classifier = nn.Sequential(
                                        nn.Linear(in_features=1000, out_features=100, bias=True), 
                                        nn.GELU(), 
                                        nn.Dropout1d(0.1), 
                                        nn.Linear(in_features=100, out_features=num_classes, bias=True)
        )

    def forward(self, sef):

        sef = sef.squeeze(1)
        
        sef = self.se_fc(sef)
        
        o = sef * sef
        
        o = self.drop(o)
        o = self.sumpool(o)

        o = torch.nn.functional.normalize(o, p = 0.5)

        o = torch.nn.functional.normalize(o, p = 2)

        return self.classifier(o)

###################################################################################################

class Multi_Modal_MFB_GCN(nn.Module):
    def __init__(self, num_classes, device):
        super(Multi_Modal_MFB_GCN, self).__init__()
        
        self.se_fc = nn.Linear(1024, 5000)
        
        self.drop = nn.Dropout1d(0.1)
        
        self.sumpool = nn.AvgPool1d(5)

        self.classifier = nn.Sequential(
                                        nn.Linear(in_features=1000, out_features=200, bias=True), 
                                        nn.GELU(), 
                                        nn.Dropout1d(0.1), 
                                        nn.Linear(in_features=200, out_features=100, bias=True)
        )
        
        #######
        
        self.gcn1 = GraphConvolution(num_classes, 50)
        self.gcn2 = GraphConvolution(50, 100)
        self.relu = nn.LeakyReLU(0.2)
        
        self.g_input = F.one_hot(torch.arange(0, num_classes), num_classes=num_classes).to(dtype=torch.float, device=device)
        g_adj = torch.ones(num_classes, num_classes)
        
        _adj = gen_A(num_classes, 0, g_adj)
        #self.A = nn.Parameter(torch.from_numpy(_adj)).to(dtype=torch.float, device=device)
        self.A = nn.Parameter(_adj).to(dtype=torch.float, device=device)

        #######
        
        self.device = device
        self.num_classes = num_classes

    def forward(self, sef):

        sef = sef.squeeze(1)
        
        sef = self.se_fc(sef)
        
        o = sef * sef
        
        o = self.drop(o)
        o = self.sumpool(o)

        o = torch.nn.functional.normalize(o, p = 0.5)

        o = torch.nn.functional.normalize(o, p = 2)
        
        #######
        

        
        # g = F.relu(self.gcn1(g_input, g_adj))
        # g = F.dropout(g, 0.1)
        # g = self.gcn2(g, g_adj)

        #######
        
        inp = self.g_input
        adj = gen_adj(self.A).detach()
        x = self.gcn1(inp, adj)
        x = self.relu(x)
        x = self.gcn2(x, adj)

        x = x.transpose(0, 1)

        #x = F.softmax(x)


        #######
        
        o = self.classifier(o)
        o = torch.mm(o, x)

        return o

###################################################################################################

class Multi_Modal_FC_GCN(nn.Module):
    def __init__(self, num_classes, device):
        super(Multi_Modal_FC_GCN, self).__init__()
        
        

        self.classifier = nn.Sequential(
                                        nn.Linear(in_features=1024, out_features=200, bias=True), 
                                        nn.GELU(), 
                                        nn.Dropout1d(0.1), 
                                        nn.Linear(in_features=200, out_features=100, bias=True)
        )
        
        #######
        
        self.gcn1 = GraphConvolution(num_classes, 50)
        self.gcn2 = GraphConvolution(50, 100)
        self.relu = nn.LeakyReLU(0.2)
        
        self.g_input = F.one_hot(torch.arange(0, num_classes), num_classes=num_classes).to(dtype=torch.float, device=device)
        g_adj = torch.ones(num_classes, num_classes)
        
        _adj = gen_A(num_classes, 0, g_adj)
        #self.A = nn.Parameter(torch.from_numpy(_adj)).to(dtype=torch.float, device=device)
        self.A = nn.Parameter(_adj).to(dtype=torch.float, device=device)

        #######
        
        self.device = device
        self.num_classes = num_classes

    def forward(self, sef):

        o = sef.squeeze(1)

        
        #######
        

        
        # g = F.relu(self.gcn1(g_input, g_adj))
        # g = F.dropout(g, 0.1)
        # g = self.gcn2(g, g_adj)

        #######
        
        inp = self.g_input
        adj = gen_adj(self.A).detach()
        x = self.gcn1(inp, adj)
        x = self.relu(x)
        x = self.gcn2(x, adj)

        x = x.transpose(0, 1)

        #x = F.softmax(x)


        #######
        
        o = self.classifier(o)
        o = torch.mm(o, x)

        return o

###################################################################################################

class Multi_Modal_FC(nn.Module):
    def __init__(self, num_classes, device):
        super(Multi_Modal_FC, self).__init__()
        
        

        self.classifier = nn.Sequential(
                                        nn.Linear(in_features=1024, out_features=200, bias=True), 
                                        nn.GELU(), 
                                        nn.Dropout1d(0.1), 
                                        nn.Linear(in_features=200, out_features=num_classes, bias=True)
        )
        
        #######
        
        self.gcn1 = GraphConvolution(num_classes, 50)
        self.gcn2 = GraphConvolution(50, 100)
        self.relu = nn.LeakyReLU(0.2)
        
        self.g_input = F.one_hot(torch.arange(0, num_classes), num_classes=num_classes).to(dtype=torch.float, device=device)
        g_adj = torch.ones(num_classes, num_classes)
        
        _adj = gen_A(num_classes, 0, g_adj)
        #self.A = nn.Parameter(torch.from_numpy(_adj)).to(dtype=torch.float, device=device)
        self.A = nn.Parameter(_adj).to(dtype=torch.float, device=device)

        #######
        
        self.device = device
        self.num_classes = num_classes

    def forward(self, sef):

        o = sef.squeeze(1)

        
        #######
        

        
        # g = F.relu(self.gcn1(g_input, g_adj))
        # g = F.dropout(g, 0.1)
        # g = self.gcn2(g, g_adj)

        #######
        
        # inp = self.g_input
        # adj = gen_adj(self.A).detach()
        # x = self.gcn1(inp, adj)
        # x = self.relu(x)
        # x = self.gcn2(x, adj)

        # x = x.transpose(0, 1)

        #x = F.softmax(x)


        #######
        
        o = self.classifier(o)
        #o = torch.mm(o, x)

        return o

###################################################################################################

