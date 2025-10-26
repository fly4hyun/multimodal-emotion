


import timm
from transformers import ElectraForSequenceClassification
from transformers import AutoConfig, Wav2Vec2FeatureExtractor

import torch
from torch import nn
import torch.nn.functional as F
import torchvision
import torchaudio

from custom_googlenet import GoogLeNet
from custom_septr import SeparableTr
from transformers import Wav2Vec2ForCTC, ASTModel


class Audio_model(nn.Module):
    def __init__(self, num_classes, device):
        super(Audio_model, self).__init__()
        
        self.device = device
        
        # # self.conv1d_1 = nn.Conv1d(1, 1, kernel_size = 7, stride = 1, padding = 3)
        self.mfcc = torchaudio.transforms.MFCC()
        # self.spec = torchaudio.transforms.Spectrogram()
        
        self.emformer_mfcc = torchaudio.models.Emformer(input_dim = 801, 
                                                        num_heads = 9, 
                                                        ffn_dim = 2048, 
                                                        num_layers = 20, 
                                                        segment_length = 4)
        #self.conv1d_mfcc = nn.Conv1d(40, 10, kernel_size = 1, stride = 1, padding = 0)
        
        # self.emformer_spec = torchaudio.models.Emformer(input_dim = 376, 
        #                                                 num_heads = 8, 
        #                                                 ffn_dim = 2048, 
        #                                                 num_layers = 10, 
        #                                                 segment_length = 4)
        # self.conv1d_spec = nn.Conv1d(201, 10, kernel_size = 1, stride = 1, padding = 0)
        # self.conv1d_2 = nn.Conv1d(1, 1, kernel_size = 7, stride = 1, padding = 3)
        
        # model_name_or_path = "harshit345/xlsr-wav2vec-speech-emotion-recognition"
        # self.wav2vec_sign = Wav2Vec2FeatureExtractor.from_pretrained(model_name_or_path)
        # # self.fc_sign = nn.Linear(16000, 20 * 81, bias = True)
        
        # self.googlenet = GoogLeNet(aux_logits = False, num_classes = 10 * 376)

        #20 * 81 + 20 * 81 + 16000
        self.classifier = nn.Sequential(nn.Linear(1 * 10 * 801, 1000, bias = True), 
                                        nn.ReLU(inplace = False), 
                                        nn.Dropout(p = 0.1, inplace = False), 
                                        nn.Linear(1000, 100, bias = True), 
                                        nn.ReLU(inplace = False), 
                                        nn.Dropout(p = 0.1, inplace = False), 
                                        nn.Linear(100, num_classes, bias = True), 
                                        #nn.Sigmoid()
        )
        
    def forward(self, x):   #   B x 16000

        #x = x.unsqueeze(1)
        x_mfcc = self.mfcc(x)#.squeeze(1)   #   B x 40 x 6
        #print(x_mfcc.shape)
        # x_spec = self.spec(x).squeeze(1)   #   B x 201 x 6
        # # x_mfcc = x_mfcc / x_mfcc.max(2)[0].unsqueeze(2)
        # # x_spec = x_spec / x_spec.max(2)[0].unsqueeze(2)
        # # x = 2 * (x - x.min(1)[0].unsqueeze(1)) / (x.max(1)[0].unsqueeze(1) - x.min(1)[0].unsqueeze(1)) - 1
        
        #x_mfcc = F.relu(self.conv1d_mfcc(x_mfcc))   #   B x 20 x 81
        x_mfcc = self.mfcc_FeatureExtraction(x_mfcc)    #   B x 20 x 81
        x_mfcc = x_mfcc.reshape(-1, 40 * 801)
        

        # x_spec = F.relu(self.conv1d_spec(x_spec))   #   B x 20 x 81
        # x_spec = self.spec_FeatureExtraction(x_spec)    #   B x 20 x 81
        # x_spec = x_spec.reshape(-1, 10 * 376)

        # # #x_sign = self.wav2vec_sign(x, sampling_rate = 16000, return_tensors = "pt", padding = True)
        # # #x_sign = x_sign['input_values'][0].to(self.device)
        # # #x_sign = self.fc_sign(x_sign)

        # x_sign = self.googlenet(x)
        

        #x = F.relu(torch.cat([x_mfcc], dim = -1))
        # x = F.relu(torch.cat([x_sign, x_mfcc, x_spec], dim = -1))

        return self.classifier(x)
    
    def mfcc_FeatureExtraction(self, x):
        
        l = torch.randint(1, 200, (x.size(0), )).cuda()
        x, l = self.emformer_mfcc(x, l)
        
        return x
    
    def spec_FeatureExtraction(self, x):
        
        l = torch.randint(1, 200, (x.size(0), )).cuda()
        x, l = self.emformer_spec(x, l)
        
        return x



class asdfasdfasdfasdf(nn.Module):
    def __init__(self, num_classes, device):
        super(Emformer_mfcc, self).__init__()
        
        self.device = device
        
        self.mfcc = torchaudio.transforms.MFCC()
        
        self.emformer_mfcc = torchaudio.models.Emformer(input_dim = 801, 
                                                        num_heads = 9, 
                                                        ffn_dim = 2048, 
                                                        num_layers = 20, 
                                                        segment_length = 4)

        self.classifier = nn.Sequential(nn.Linear(40 * 801, 1000, bias = True), 
                                        nn.ReLU(inplace = False), 
                                        nn.Dropout(p = 0.1, inplace = False), 
                                        nn.Linear(1000, 100, bias = True), 
                                        nn.ReLU(inplace = False), 
                                        nn.Dropout(p = 0.1, inplace = False), 
                                        nn.Linear(100, num_classes, bias = True), 
                                        #nn.Sigmoid()
        )
        
    def forward(self, x):   #   B x 16000

        x = self.mfcc(x)   #   B x 40 x 6
        
        x = self.mfcc_FeatureExtraction(x)    #   B x 20 x 81
        x = x.reshape(-1, 40 * 801)

        return self.classifier(x)
    
    def mfcc_FeatureExtraction(self, x):
        
        if self.training:
            l = torch.randint(1, 200, (x.size(0), )).cuda()
            x, l = self.emformer_mfcc(x, l)
        else:
            l = torch.ones(x.size(0)) * x.size(1).cuda()
            x, l = self.emformer_mfcc.infer(x, l)
        
        return x


####################################################################################################
    
class Emformer_mfcc(nn.Module):
    def __init__(self, num_classes, device):
        super(Emformer_mfcc, self).__init__()
        
        self.device = device
        
        self.mfcc = torchaudio.transforms.MFCC()
        
        self.emformer_mfcc = torchaudio.models.Emformer(input_dim = 801, 
                                                        num_heads = 9, 
                                                        ffn_dim = 2048, 
                                                        num_layers = 20, 
                                                        segment_length = 4)

        self.classifier = nn.Sequential(nn.Linear(40 * 801, 1000, bias = True), 
                                        nn.ReLU(inplace = False), 
                                        nn.Dropout(p = 0.1, inplace = False), 
                                        nn.Linear(1000, 100, bias = True), 
                                        nn.ReLU(inplace = False), 
                                        nn.Dropout(p = 0.1, inplace = False), 
                                        nn.Linear(100, num_classes, bias = True), 
                                        #nn.Sigmoid()
        )
        
    def forward(self, x):   #   B x 16000

        x = self.mfcc(x)   #   B x 40 x 6
        
        x = self.mfcc_FeatureExtraction(x)    #   B x 20 x 81
        x = x.reshape(-1, 40 * 801)

        return self.classifier(x)
    
    def mfcc_FeatureExtraction(self, x):
        
        l = torch.randint(1, 200, (x.size(0), )).to(self.device)
        x, l = self.emformer_mfcc(x, l)
        
        return x

####################################################################################################

class Transformer_Model(nn.Module):
    def __init__(self, num_classes, device):
        super(Transformer_Model, self).__init__()
        
        self.device = device
        
        self.fc = nn.Linear(313, 1024, bias = True)
        transformer_encoder = [nn.TransformerEncoderLayer(d_model=1024, nhead=8) for _ in range(4)]
        
        self.encoder_layer_10 = nn.Sequential(*transformer_encoder)
        #self.encoder_layer_11 = nn.TransformerEncoderLayer(d_model=1024, nhead=8)
        #self.encoder_layer_12 = nn.TransformerEncoderLayer(d_model=1024, nhead=8)

        self.classifier = nn.Sequential(nn.Linear(1024, 256, bias = True), 
                                        nn.ReLU(inplace = False), 
                                        nn.Dropout(p = 0.2, inplace = False), 
                                        nn.Linear(256, num_classes, bias = True), 
                                        #nn.Sigmoid()
        )
        
    def forward(self, x):   #   B x 160000
        
        x_mfcc = x
        positional_encoding = self.Positional_Encoding()
        
        x_mfcc = x_mfcc + positional_encoding[:x_mfcc.size(1), :x_mfcc.size(2)]
        x_mfcc = self.fc(x_mfcc)
        
        x_10 = self.encoder_layer_10(x_mfcc)
        #x_11 = self.encoder_layer_11(x_10)
        #x_12 = self.encoder_layer_12(x_11)
        
        x = x_10[:, -1, :]# + x_11[:, -1, :] + x_12[:, -1, :]
        
        return self.classifier(x)
    
    def Positional_Encoding(self):
        
        max_len = 50
        d_model = 810
        
        encoding = torch.zeros(max_len, d_model, device = self.device)
        encoding.requires_grad = False

        pos = torch.arange(0, max_len, device =self.device)
        pos = pos.float().unsqueeze(dim=1)
        
        _2i = torch.arange(0, d_model, step=2, device=self.device).float()
        
        encoding[:, ::2] = torch.sin(pos / (10000 ** (_2i / d_model)))
        encoding[:, 1::2] = torch.cos(pos / (10000 ** (_2i / d_model)))
        
        return encoding
        
####################################################################################################
        
class GoogleNet_1d(nn.Module):
    def __init__(self, num_classes, device):
        super(GoogleNet_1d, self).__init__()
        
        self.device = device
        
        self.googlenet = GoogLeNet(aux_logits = False, num_classes = 1000)

        self.classifier = nn.Sequential(nn.Linear(1000, 100, bias = True), 
                                        nn.ReLU(inplace = False), 
                                        nn.Dropout(p = 0.1, inplace = False), 
                                        nn.Linear(100, num_classes, bias = True), 
                                        #nn.Sigmoid()
        )
        
    def forward(self, x):   #   B x 16000

        x = self.googlenet(x.unsqueeze(1))

        return self.classifier(x)


####################################################################################################3


class Temporal_Aware_Block(nn.Module):
    def __init__(self, x_size_1, x_size_2, i):
        super(Temporal_Aware_Block, self).__init__()
        
        self.conv1_1 = nn.Conv1d(x_size_1, x_size_1, kernel_size = 2, dilation = i, padding = 0)
        self.conv2_1 = nn.Conv1d(x_size_1, x_size_1, kernel_size = 2, dilation = i, padding = 0)
        self.batch_norm_1 = nn.BatchNorm1d(num_features = x_size_1)
        self.batch_norm_2 = nn.BatchNorm1d(num_features = x_size_1)
        self.relu = F.relu
        self.dropout_1 = nn.Dropout(0.1)
        self.dropout_2 = nn.Dropout(0.1)
        
        self.conv1 = nn.Conv1d(x_size_1, x_size_1, kernel_size = 1, padding = 0)
        
    def forward(self, x):
        
        x_ori = x

        x = nn.functional.pad(x, (self.conv1_1.dilation[0], 0))
        # x_1 = self.dropout(self.relu(self.batch_norm(self.conv1_1(x))))
        
        #padding = (x.shape[2] - 1) * self.conv1_1.dilation[0] - x.shape[2] + self.conv1_1.kernel_size[0]
        #x = F.pad(x, (padding, 0))
        conv_1_1 = self.conv1_1(x)
        conv_1_1 = self.batch_norm_1(conv_1_1)
        conv_1_1 = self.relu(conv_1_1)
        output_1_1 = self.dropout_1(conv_1_1)
        
        output_1_1 = nn.functional.pad(output_1_1, (self.conv2_1.dilation[0], 0))
        # x_2 = self.dropout(self.relu(self.batch_norm(self.conv2_1(x_1))))
        # x_2 = self.sigmoid(x_2)
        
        #padding = (output_1_1.shape[2] - 1) * self.conv2_1.dilation[0] - output_1_1.shape[2] + self.conv2_1.kernel_size[0]
        #output_1_1 = F.pad(output_1_1, (padding, 0))
        conv_2_1 = self.conv2_1(output_1_1)
        conv_2_1 = self.batch_norm_2(conv_2_1)
        conv_2_1 = self.relu(conv_2_1)
        output_2_1 = self.dropout_2(conv_2_1)
        
        if x_ori.shape[-1] != output_2_1.shape[-1]:
            x_ori = self.conv1(x_ori)
        
        output_2_1 = torch.sigmoid(output_2_1)
        F_x = torch.mul(x_ori, output_2_1)
        
        return F_x

class TIMNet(nn.Module):
    def __init__(self, x_size_1, x_size_2, num_classes, device):
        super(TIMNet, self).__init__()
        
        self.device = device
        self.num_skip_layer = 10
        
        self.x_size_1 = x_size_1
        self.x_size_2 = x_size_2
        
        self.conv1d_f = nn.Conv1d(self.x_size_1, self.x_size_1, kernel_size = 1, dilation = 1, padding = 0)
        self.conv1d_b = nn.Conv1d(self.x_size_1, self.x_size_1, kernel_size = 1, dilation = 1, padding = 0)
        
        self.skip_model = []
        self.skip_backward_model = []
        for i in [2 ** i for i in range(self.num_skip_layer)]:
            self.skip_model.append(Temporal_Aware_Block(self.x_size_1, self.x_size_2, i))#.to(self.device))
        
        self.skip_model = nn.ModuleList(self.skip_model)
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        
        
        self.dynamic_fusion = nn.Parameter(torch.ones(self.num_skip_layer, 1)).to(self.device)
        
        self.classifier = nn.Sequential(nn.Linear(self.x_size_1, num_classes)
                                        #nn.Sigmoid()
        )
        
    def forward(self, x):   #   B x 160000
        
        x_forward = x
        x_backward = torch.flip(x, [2])
        
        skip_out_forward = self.conv1d_f(x_forward)
        skip_out_backward = self.conv1d_b(x_backward)
        
        final_skip_concat = []
        for i in range(self.num_skip_layer):
            skip_out_forward = self.skip_model[i](skip_out_forward)
            skip_out_backward = self.skip_model[i](skip_out_backward)
            
            temp_skip = skip_out_forward + skip_out_backward
            temp_skip = self.global_avg_pool(temp_skip)
            final_skip_concat.append(temp_skip.transpose(1, 2))

        
        output_2 = final_skip_concat[0]

        for i, item in enumerate(final_skip_concat):
            if i == 0:
                continue
            output_2 = torch.cat([output_2, item], dim=-2)
        x = output_2
        o = (x * self.dynamic_fusion).sum(1)
            
        return self.classifier(o)
            
###################################################################################################

class Xlsr(nn.Module):
    def __init__(self, num_classes, device):
        super(Xlsr, self).__init__()
        
        self.device = device
        
        MODEL_ID = "jonatasgrosman/wav2vec2-large-xlsr-53-english"

        self.model = Wav2Vec2ForCTC.from_pretrained(MODEL_ID)
        #self.model.lm_head = nn.Linear(in_features=1024, out_features=num_classes, bias=True)
        classifier = [
            nn.Linear(33, num_classes)]
        
        self.classifier = nn.Sequential(*classifier)
        
    def forward(self, x, y):
        
        x = self.model(x, y).logits

        o = self.classifier(x[:, -1, :])
        
        return o