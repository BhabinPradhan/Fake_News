import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.models import resnet18
from transformers import BertModel, BertConfig, BertTokenizer, get_cosine_schedule_with_warmup
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModelForMaskedLM

import math

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
                    nn.Dropout(0.1),
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

class CLIP(nn.Module):
    def __init__(self, out_channels, bert_path, 
        cnn_channel=32,
        cnn_kernel_size=(1, 2, 4, 8),
        shared_image_dim=128,
        shared_text_dim=128):
        super(CLIP, self).__init__()
        
        # Image Encoder - Build from scratch to avoid pretrained issues
        resnet = resnet18(pretrained=False)  # Start with random weights
        
        # Use only the convolutional layers (no FC)
        self.img_conv = nn.Sequential(
            resnet.conv1,      # 3 -> 64
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,     # 64 -> 64
            resnet.layer2,     # 64 -> 128
            resnet.layer3,     # 128 -> 256
            resnet.layer4,     # 256 -> 512
        )
        
        self.img_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Simple projection with proper initialization
        self.img_projection = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, out_channels),
        )
        
        # Initialize weights properly
        for m in self.img_projection.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=1.0)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        
        # Text encoder - BERT frozen
        self.config = BertConfig.from_pretrained(bert_path)
        self.bert = BertModel.from_pretrained(bert_path)
        
        # Freeze BERT by default
        for param in self.bert.parameters():
            param.requires_grad = False
        # Unfreeze last 2 encoder layers + pooler (cross-modal alignment benefit)
        for name, param in self.bert.named_parameters():
            if "encoder.layer.10" in name or "encoder.layer.11" in name or "pooler" in name:
                param.requires_grad = True
            
        # Text CNN
        self.text_cnn = FastCNN(
            channel=cnn_channel,
            kernel_size=cnn_kernel_size
        )
        
        # Text projection
        self.text_projection = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, out_channels),
        )
        
        # Initialize text projection
        for m in self.text_projection.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=1.0)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, input_ids, attention_mask, token_type_ids, img):
        batch_size = img.size(0)
        
        # === IMAGE PATH ===
        # Extract features
        img_features = self.img_conv(img)  # [B, 512, H, W]
        img_pooled = self.img_pool(img_features)  # [B, 512, 1, 1]
        img_flat = img_pooled.view(batch_size, -1)  # [B, 512]
        
        # Project to output space
        img_out = self.img_projection(img_flat)  # [B, out_channels]
        
        # === TEXT PATH ===
        bert_output = self.bert(input_ids, attention_mask, token_type_ids)
        text_encoding = bert_output[0]  # [B, seq_len, 768]
        
        # CNN features
        text_features = self.text_cnn(text_encoding)  # [B, 128]
        
        # Project to output space
        text_out = self.text_projection(text_features)  # [B, out_channels]

        return img_out, text_out

    def encode_image(self, image):
        batch_size = image.size(0)
        img_features = self.img_conv(image)
        img_pooled = self.img_pool(img_features)
        img_flat = img_pooled.view(batch_size, -1)
        img_out = self.img_projection(img_flat)
        return img_out

    def encode_text(self, input_ids, attention_mask, token_type_ids):
        with torch.no_grad():
            bert_output = self.bert(input_ids, attention_mask, token_type_ids)
            text_encoding = bert_output[0]
        
        text_features = self.text_cnn(text_encoding)
        text_out = self.text_projection(text_features)
        return text_out
