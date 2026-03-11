import torch
import torch.nn as nn
import torch.nn.functional as F
import jieba
import logging
from transformers import BertModel
from torchvision import models

# Silence jieba startup logs
jieba.setLogLevel(logging.ERROR)

class ImprovedDynamicRouting(nn.Module):
    """Implementing Algorithm 1: Alignment of Visual Entities to Textual Entities"""
    def __init__(self, dim, iterations=3):
        super().__init__()
        self.iterations = iterations
        self.dim = dim

    def squash(self, x):
        norm = torch.norm(x, dim=-1, keepdim=True)
        return (norm / (1 + norm**2 + 1e-8)) * x

    def forward(self, textual_entities, visual_entities):
        """
        textual_entities: [batch, 100, 768]
        visual_entities:  [batch, 49, 768]
        """
        batch_size = textual_entities.size(0)
        n = textual_entities.size(1) # 100
        m = visual_entities.size(1)  # 49

        # b_ij: Initial coupling coefficients
        b = torch.zeros(batch_size, n, m).to(textual_entities.device)

        for i in range(self.iterations):
            # c: Softmax over visual entities
            c = torch.softmax(b, dim=-1) # [batch, 100, 49]
            
            # ALIGN: Create visual summary for each text entity
            aligned_ve = torch.bmm(c, visual_entities) # [batch, 100, 768]
            aligned_ve = self.squash(aligned_ve)

            if i < self.iterations - 1:
                # UPDATE: Agreement (Similarity)
                agreement = torch.bmm(textual_entities, visual_entities.transpose(1, 2))
                b = b + agreement

        return aligned_ve 

class CrossModalFusion(nn.Module):
    """Implementing Section III-C: Attend, Compare, and Aggregate"""
    def __init__(self, dim):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(dim, num_heads=8, batch_first=True)
        self.compare_fc = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.ReLU()
        )

    def forward(self, te, aligned_ve):
        te_attn, _ = self.self_attn(te, te, te)
        
        diff = torch.abs(te_attn - aligned_ve)
        sim = te_attn * aligned_ve
        
        compare_feat = torch.cat([diff, sim], dim=-1) 
        compare_feat = self.compare_fc(compare_feat)

        avg_pool = torch.mean(compare_feat, dim=1)
        max_pool, _ = torch.max(compare_feat, dim=1)
        
        return torch.cat([avg_pool, max_pool], dim=-1)

class EMAF_Model(nn.Module):
    def __init__(self, bert_path):
        super(EMAF_Model, self).__init__()
        self.bert = BertModel.from_pretrained(bert_path)
        
        resnet = models.resnet50(pretrained=True)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2]) 
        self.img_proj = nn.Linear(2048, 768)

        self.alignment = ImprovedDynamicRouting(dim=768)
        self.fusion = CrossModalFusion(dim=768)
        
        self.classifier = nn.Sequential(
            nn.Linear(768 * 2, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 2)
        )

    def forward(self, input_ids, attention_mask, images):
        batch_size, n_entities, seq_len = input_ids.shape

        # --- FIX: FLATTEN FOR BERT ---
        # Flatten [Batch, 100, 10] -> [Batch*100, 10]
        flat_ids = input_ids.view(-1, seq_len)
        flat_mask = attention_mask.view(-1, seq_len)

        text_out = self.bert(input_ids=flat_ids, attention_mask=flat_mask)
        # Reshape back to [Batch, 100, 768]
        te = text_out.last_hidden_state[:, 0, :].view(batch_size, n_entities, -1) 

        # --- VISUAL ---
        grid_feats = self.backbone(images) 
        ve = grid_feats.view(grid_feats.size(0), 2048, -1).transpose(1, 2) 
        ve = self.img_proj(ve) # [Batch, 49, 768]

        # --- ALIGN & FUSE ---
        aligned_ve = self.alignment(te, ve)
        fused_vector = self.fusion(te, aligned_ve)

        return self.classifier(fused_vector)