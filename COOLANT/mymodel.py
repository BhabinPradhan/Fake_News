from distutils.command.config import config
import pandas as pd 
import numpy as np 
import json, time 
from tqdm import tqdm 
from sklearn.metrics import accuracy_score, classification_report
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertModel, BertConfig, BertTokenizer, get_cosine_schedule_with_warmup
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModelForMaskedLM
import warnings
import re
import torchvision
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable, Function
from torch.utils.data import Dataset
import math
import random
from re import X
import torch
import torch.nn as nn
from torch.distributions import Normal, Independent
from torch.nn.functional import softplus
from torchvision.models import resnet18
from SENet import Network

class Encoder(nn.Module):
    def __init__(self, z_dim=2):
        super(Encoder, self).__init__()
        self.z_dim = z_dim
        self.net = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(True),
            nn.Linear(64, z_dim * 2),
        )

    def forward(self, x):
        params = self.net(x)
        mu, sigma = params[:, :self.z_dim], params[:, self.z_dim:]
        sigma = softplus(sigma) + 1e-7 
        return Independent(Normal(loc=mu, scale=sigma), 1)

class AmbiguityLearning(nn.Module):
    def __init__(self):
        super(AmbiguityLearning, self).__init__()
        self.encoder_text = Encoder()
        self.encoder_image = Encoder()

    def forward(self, text_encoding, image_encoding):
        p_z1_given_text = self.encoder_text(text_encoding) 
        p_z2_given_image = self.encoder_image(image_encoding)
        z1 = p_z1_given_text.rsample() 
        z2 = p_z2_given_image.rsample()
        kl_1_2 = p_z1_given_text.log_prob(z1) - p_z2_given_image.log_prob(z1)
        kl_2_1 = p_z2_given_image.log_prob(z2) - p_z1_given_text.log_prob(z2)
        skl = (kl_1_2 + kl_2_1) / 2.
        skl = torch.sigmoid(skl)
        return skl

class UnimodalDetection(nn.Module):
    def __init__(self, shared_dim=128, prime_dim=16):
        super(UnimodalDetection, self).__init__()
        # FIX: Remove BatchNorm - it's causing feature collapse
        self.text_uni = nn.Sequential(
            nn.Linear(shared_dim, shared_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(shared_dim, prime_dim),
            nn.ReLU()
        )
        self.image_uni = nn.Sequential(
            nn.Linear(shared_dim, shared_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(shared_dim, prime_dim),
            nn.ReLU()
        )

    def forward(self, text_encoding, image_encoding):
        text_prime = self.text_uni(text_encoding)
        image_prime = self.image_uni(image_encoding)
        return text_prime, image_prime

class CrossModule4Batch(nn.Module):
    def __init__(self, text_in_dim=64, image_in_dim=64, corre_out_dim=64):
        super(CrossModule4Batch, self).__init__()
        self.softmax = nn.Softmax(-1)
        self.corre_dim = 64
        self.pooling = nn.AdaptiveMaxPool1d(1)
        self.c_specific_2 = nn.Sequential(
            nn.Linear(self.corre_dim, corre_out_dim),
            nn.ReLU()
        )

    def forward(self, text, image):
        text_in = text.unsqueeze(2) 
        image_in = image.unsqueeze(1) 
        corre_dim = text.shape[1]
        similarity = torch.matmul(text_in, image_in) / math.sqrt(corre_dim)
        correlation = self.softmax(similarity)
        correlation_p = self.pooling(correlation).squeeze()
        
        # Handle edge case where squeeze removes batch dim
        if correlation_p.dim() == 1:
            correlation_p = correlation_p.unsqueeze(0)
            
        correlation_out = self.c_specific_2(correlation_p)
        return correlation_out 

class FastCNN(nn.Module):
    def __init__(self, channel=32, kernel_size=(1, 2, 4, 8)):
        super(FastCNN, self).__init__()
        self.fast_cnn = nn.ModuleList()
        for kernel in kernel_size:
            self.fast_cnn.append(
                nn.Sequential(
                    nn.Conv1d(768, channel, kernel_size=kernel),
                    nn.BatchNorm1d(channel),
                    nn.ReLU(),
                    nn.Dropout(),
                    nn.AdaptiveMaxPool1d(1) 
                )
            )

    def forward(self, x):
        x = x.permute(0, 2, 1) 
        x_out = []
        for module in self.fast_cnn:
            x_out.append(module(x).squeeze(-1)) 
        x_out = torch.cat(x_out, 1)
        return x_out 

class EncodingPart(nn.Module):
    def __init__(
        self,
        cnn_channel=32,
        cnn_kernel_size=(1, 2, 4, 8),
        shared_image_dim=128,
        shared_text_dim=128
    ):
        super(EncodingPart, self).__init__()
        self.shared_text_encoding = FastCNN(
            channel=cnn_channel,
            kernel_size=cnn_kernel_size
        )
        self.shared_text_linear = nn.Sequential(
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(64, shared_text_dim),
            nn.BatchNorm1d(shared_text_dim),
            nn.ReLU()
        )
        self.shared_image = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(256, shared_image_dim),
            nn.BatchNorm1d(shared_image_dim),
            nn.ReLU()
        )

    def forward(self, text, image):
        text_encoding = self.shared_text_encoding(text) 
        text_shared = self.shared_text_linear(text_encoding)
        image_shared = self.shared_image(image) 
        return text_shared, image_shared

class SimilarityModule(nn.Module):
    def __init__(self, bert_path, image_fea=512, shared_dim=128, sim_dim=64):
        super(SimilarityModule, self).__init__()
        self.bert = BertModel.from_pretrained(bert_path)
        
        for param in self.bert.parameters():
            param.requires_grad = False
        for name, param in self.bert.named_parameters():
            if "encoder.layer.11" in name or "pooler" in name:
                param.requires_grad = True

        res18 = resnet18(pretrained=True)
        self.img_model = nn.Sequential(*list(res18.children())[:-1])
        self.img_fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, image_fea),
            nn.ReLU()
        )

        self.encoding = EncodingPart()
        self.text_aligner = nn.Sequential(
            nn.Linear(shared_dim, sim_dim), 
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.image_aligner = nn.Sequential(
            nn.Linear(shared_dim, sim_dim), 
            nn.ReLU(),
            nn.Dropout(0.2)
        )

    def forward(self, input_ids, attention_mask, token_type_ids, img):
        outputs = self.bert(input_ids, attention_mask, token_type_ids)
        text = outputs[0]
        
        # FIX: Don't normalize - preserve variance
        img_features = self.img_model(img)
        img_out = self.img_fc(img_features)
        
        text_enc, img_enc = self.encoding(text, img_out)
        return self.text_aligner(text_enc), self.image_aligner(img_enc), None

class Multi_Model(nn.Module):
    def __init__(self, bert_path, feature_dim=64+16+16, h_dim=128):
        super(Multi_Model, self).__init__()
        self.bert = BertModel.from_pretrained(bert_path)
        
        res18 = resnet18(pretrained=True)
        self.image_backbone = nn.Sequential(*list(res18.children())[:-1])
        
        # FIX: Don't freeze ResNet - let it adapt
        for param in self.image_backbone.parameters():
            param.requires_grad = True
        
        # Simpler feature extractors
        self.image_fc = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128)
        )
        self.text_fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128)
        )

        self.uni_repre = UnimodalDetection(shared_dim=128, prime_dim=16)
        self.uni_se = UnimodalDetection(shared_dim=128, prime_dim=64)
        self.cross_module = CrossModule4Batch()
        self.senet = Network(64, 128, 24, 3)
        self.ambiguity_module = AmbiguityLearning()
        
        # Simpler classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(feature_dim, h_dim),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(h_dim, 2)
        )

    def forward(self, input_ids, attention_mask, token_type_ids, img, t_aln, i_aln):
        # Extract base features WITHOUT over-normalization
        bert_output = self.bert(input_ids, attention_mask, token_type_ids)[1]
        t_feat = self.text_fc(bert_output)
        
        img_features = self.image_backbone(img)
        i_feat = self.image_fc(img_features)
        
        # Process through unimodal paths
        t_prime, i_prime = self.uni_repre(t_feat, i_feat)
        t_se, i_se = self.uni_se(t_feat, i_feat)
        corr = self.cross_module(t_aln, i_aln)

        # Get attention from SENet
        attn = self.senet(torch.cat([
            t_se.unsqueeze(-1), 
            i_se.unsqueeze(-1), 
            corr.unsqueeze(-1)
        ], -1))
        
        # Apply attention with sigmoid (keeps variance)
        attn_weights = torch.sigmoid(attn)
        
        t_weighted = t_prime * attn_weights[:,0].unsqueeze(1)
        i_weighted = i_prime * attn_weights[:,1].unsqueeze(1)
        c_weighted = corr * attn_weights[:,2].unsqueeze(1)
        
        # Concatenate
        final = torch.cat([t_weighted, i_weighted, c_weighted], 1)
        
        # Classify
        output = self.classifier(final)
        
        return output, attn, self.ambiguity_module(t_aln, i_aln)
