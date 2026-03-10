import warnings
warnings.filterwarnings('ignore')
import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.nn.utils as utils
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import torchvision.models as models
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torch.utils.data import TensorDataset
from sklearn.metrics import f1_score
import torch.nn.init as init
import pickle
import abc
import json, os
import argparse
import random
from PIL import Image
import sys
from torch.distributions import Normal, Independent
from torch.nn.functional import softplus
from tqdm import tqdm
import pandas as pd
import ast
from transformers import BertTokenizer, BertModel

# Add argument parser for fold number
parser = argparse.ArgumentParser()
parser.add_argument('--fold', type=int, default=0, help='Fold number (0-4) for k-fold cross validation')
args, _ = parser.parse_known_args()

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    def __init__(self, fold=0):
        self.base_dir = "/home/odobasia/Downloads/BERT_Project/Med-MMHL"
        self.img_dir = os.path.join(self.base_dir, "all_medical_images")
        
        # Matches your screenshot: data/image_article/train_fold0.csv, etc.
        self.train_csv = os.path.join(self.base_dir, f"data/image_article/train_fold{fold}.csv")
        self.test_csv = os.path.join(self.base_dir, f"data/image_article/test_fold{fold}.csv")
        
        # Static dev file for validation during training
        self.dev_csv = os.path.join(self.base_dir, "data/image_article/dev.csv")
        
        self.maxlen = 170
        self.num_classes = 2
        self.target_names = ['Real', 'Fake']
        self.kernel_sizes = [1, 2, 3, 5]
        self.batch_size = 8
        self.epochs = 30
        self.save_path = f'./best_moped_mmhl_fold{fold}.pth'
        
        # Architecture features (keep these the same)
        self.fc5_in_features = 3448
        self.fc4_in_features = 1024
        self.fc3_in_features = 512
        self.fc2_in_features = 500
        self.fc1_in_features = 200
        self.kl_in_features = 440

config = Config(fold=args.fold)

# ============================================================================
# DATASET LOADER FOR MED-MMHL
# ============================================================================

class MedMMHLDataset(Dataset):
    def __init__(self, csv_path, img_base_dir, tokenizer, max_length=170, is_training=False):
        self.data = pd.read_csv(csv_path)
        self.img_base_dir = img_base_dir
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.is_training = is_training
        
        # Image transforms
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        self.train_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        # ResNet50 feature extractor
        self.resnet = models.resnet50(pretrained=True)
        self.resnet.eval()
        for param in self.resnet.parameters():
            param.requires_grad = False
        # Remove final FC layer to get features
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-1])
        
        if torch.cuda.is_available():
            self.resnet = self.resnet.cuda()

    def _clean_path(self, path_str):
        """Cleans CSV strings like "['../images/path/img.jpg']" into a usable relative path."""
        try:
            path_list = ast.literal_eval(path_str)
            if isinstance(path_list, list) and len(path_list) > 0:
                raw_path = path_list[0]
                return raw_path.replace('../images/', '').replace('./images/', '')
        except (ValueError, SyntaxError):
            return str(path_str).replace('../images/', '')
        return None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_rel_path = self._clean_path(row['image'])
        
        # Load image and extract ResNet features
        image_feature = torch.zeros(2048)
        if img_rel_path:
            full_path = os.path.join(self.img_base_dir, img_rel_path)
            if os.path.exists(full_path):
                try:
                    img_pil = Image.open(full_path).convert('RGB')
                    transform = self.train_transform if self.is_training else self.transform
                    img_tensor = transform(img_pil).unsqueeze(0)
                    
                    if torch.cuda.is_available():
                        img_tensor = img_tensor.cuda()
                    
                    with torch.no_grad():
                        features = self.resnet(img_tensor)
                        image_feature = features.squeeze().cpu()
                except Exception as e:
                    pass

        text = str(row['content'])
        label = int(row['det_fake_label'])

        # Tokenize text
        tokens = self.tokenizer(
            text, 
            padding='max_length', 
            truncation=True, 
            max_length=self.max_length, 
            return_tensors='pt'
        )
        
        input_ids = tokens['input_ids'].squeeze(0)
        attention_mask = tokens['attention_mask'].squeeze(0)
        
        return input_ids, attention_mask, image_feature, label

def collate_fn(batch):
    """Custom collate function for DataLoader"""
    input_ids = torch.stack([item[0] for item in batch])
    attention_masks = torch.stack([item[1] for item in batch])
    image_features = torch.stack([item[2] for item in batch])
    labels = torch.tensor([item[3] for item in batch])
    
    return input_ids, attention_masks, image_features, labels

# ============================================================================
# MODEL COMPONENTS (from MoPeD)
# ============================================================================

class PGD(object):
    def __init__(self, model, emb_name, epsilon=1., alpha=0.3):
        self.model = model
        self.emb_name = emb_name
        self.epsilon = epsilon
        self.alpha = alpha
        self.emb_backup = {}
        self.grad_backup = {}

    def attack(self, is_first_attack=False):
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.emb_name in name:
                if is_first_attack:
                    self.emb_backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0:
                    r_at = self.alpha * param.grad / norm
                    param.data.add_(r_at)
                    param.data = self.project(name, param.data, self.epsilon)

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.emb_name in name:
                assert name in self.emb_backup
                param.data = self.emb_backup[name]
        self.emb_backup = {}

    def project(self, param_name, param_data, epsilon):
        r = param_data - self.emb_backup[param_name]
        if torch.norm(r) > epsilon:
            r = epsilon * r / torch.norm(r)
        return self.emb_backup[param_name] + r

    def backup_grad(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                self.grad_backup[name] = param.grad.clone()

    def restore_grad(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                param.grad = self.grad_backup[name]

class TransformerBlock(nn.Module):
    def __init__(self, input_size, d_k=16, d_v=16, n_heads=8, is_layer_norm=False, attn_dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.n_heads = n_heads
        self.d_k = d_k if d_k is not None else input_size
        self.d_v = d_v if d_v is not None else input_size

        self.is_layer_norm = is_layer_norm
        if is_layer_norm:
            self.layer_morm = nn.LayerNorm(normalized_shape=input_size)

        self.W_q = nn.Parameter(torch.Tensor(input_size, n_heads * d_k))
        self.W_k = nn.Parameter(torch.Tensor(input_size, n_heads * d_k))
        self.W_v = nn.Parameter(torch.Tensor(input_size, n_heads * d_v))

        self.W_o = nn.Parameter(torch.Tensor(d_v*n_heads, input_size))
        self.linear1 = nn.Linear(input_size, input_size)
        self.linear2 = nn.Linear(input_size, input_size)

        self.dropout = nn.Dropout(attn_dropout)
        self.__init_weights__()

    def __init_weights__(self):
        init.xavier_normal_(self.W_q)
        init.xavier_normal_(self.W_k)
        init.xavier_normal_(self.W_v)
        init.xavier_normal_(self.W_o)
        init.xavier_normal_(self.linear1.weight)
        init.xavier_normal_(self.linear2.weight)

    def FFN(self, X):
        output = self.linear2(F.relu(self.linear1(X)))
        output = self.dropout(output)
        return output

    def scaled_dot_product_attention(self, Q, K, V, episilon=1e-6):
        temperature = self.d_k ** 0.5
        Q_K = torch.einsum("bqd,bkd->bqk", Q, K) / (temperature + episilon)
        Q_K_score = F.softmax(Q_K, dim=-1)
        Q_K_score = self.dropout(Q_K_score)

        V_att = Q_K_score.bmm(V)
        return V_att

    def multi_head_attention(self, Q, K, V):
        bsz, q_len, _ = Q.size()
        bsz, k_len, _ = K.size()
        bsz, v_len, _ = V.size()

        Q_ = Q.matmul(self.W_q).view(bsz, q_len, self.n_heads, self.d_k)
        K_ = K.matmul(self.W_k).view(bsz, k_len, self.n_heads, self.d_k)
        V_ = V.matmul(self.W_v).view(bsz, v_len, self.n_heads, self.d_v)

        Q_ = Q_.permute(0, 2, 1, 3).contiguous().view(bsz*self.n_heads, q_len, self.d_k)
        K_ = K_.permute(0, 2, 1, 3).contiguous().view(bsz*self.n_heads, q_len, self.d_k)
        V_ = V_.permute(0, 2, 1, 3).contiguous().view(bsz*self.n_heads, q_len, self.d_v)

        V_att = self.scaled_dot_product_attention(Q_, K_, V_)
        V_att = V_att.view(bsz, self.n_heads, q_len, self.d_v)
        V_att = V_att.permute(0, 2, 1, 3).contiguous().view(bsz, q_len, self.n_heads*self.d_v)

        output = self.dropout(V_att.matmul(self.W_o))
        return output

    def forward(self, Q, K, V):
        V_att = self.multi_head_attention(Q, K, V)

        if self.is_layer_norm:
            X = self.layer_morm(Q + V_att)
            output = self.layer_morm(self.FFN(X) + X)
        else:
            X = Q + V_att
            output = self.FFN(X) + X
        return output

class Encoder(nn.Module):
    def __init__(self, z_dim=2):
        super(Encoder, self).__init__()
        self.z_dim = z_dim
        self.net = nn.Sequential(
            nn.Linear(862, config.kl_in_features),
            nn.ReLU(True),
            nn.Linear(config.kl_in_features, z_dim * 2),
        )

    def forward(self, x):
        params = self.net(x)
        mu, sigma = params[:, :self.z_dim], params[:, self.z_dim:]
        sigma = softplus(sigma) + 1e-7
        return Independent(Normal(loc=mu, scale=sigma), 1)

class KL(nn.Module):
    def __init__(self):
        super(KL, self).__init__()
        self.encoder_text = Encoder()
        self.encoder_image = Encoder()

    def forward(self, text_encoding, image_encoding):
        p_z1_given_text = self.encoder_text(text_encoding)
        p_z2_given_image = self.encoder_image(image_encoding)
        z1 = p_z1_given_text.rsample()
        z2 = p_z2_given_image.rsample()
        kl_1_2 = p_z1_given_text.log_prob(z1) - p_z2_given_image.log_prob(z1)
        kl_2_1 = p_z2_given_image.log_prob(z2) - p_z1_given_text.log_prob(z2)
        skl = (kl_1_2 + kl_2_1)/ 2.
        skl = torch.sigmoid(skl)
        return skl

# ============================================================================
# MoPeD MODEL ADAPTED FOR MED-MMHL
# ============================================================================

class MoPeD_MMHL(nn.Module):
    def __init__(self, config, tokenizer):
        super(MoPeD_MMHL, self).__init__()
        self.config = config
        self.tokenizer = tokenizer
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.best_acc = 0
        
        # BERT for text encoding
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        
        # Freeze BERT layers except last 2
        for param in self.bert.parameters():
            param.requires_grad = False
        for param in self.bert.encoder.layer[-2:].parameters():
            param.requires_grad = True
        
        # Project BERT output (768) to 300
        self.text_projection = nn.Linear(768, 300)
        
        # Image feature projection (2048 -> 300)
        self.image_projection = nn.Linear(2048, 300)
        
        # Text CNN layers
        self.dropout = nn.Dropout(0.5)
        self.dropout_heavy = nn.Dropout(0.7)
        
        self.convs = nn.ModuleList([
            nn.Conv1d(300, 100, kernel_size=K) for K in config.kernel_sizes
        ])
        
        # Attention
        self.mh_attention = TransformerBlock(
            input_size=862, 
            n_heads=8, 
            attn_dropout=0.4
        )
        
        # Batch normalization
        self.bn1 = nn.BatchNorm1d(config.fc4_in_features, momentum=0.05)
        self.bn2 = nn.BatchNorm1d(config.fc3_in_features, momentum=0.05)
        self.bn3 = nn.BatchNorm1d(config.fc2_in_features, momentum=0.05)
        self.bn4 = nn.BatchNorm1d(config.fc1_in_features, momentum=0.05)
        
        # FC layers
        self.fc5 = nn.Linear(config.fc5_in_features, config.fc4_in_features)
        self.fc4 = nn.Linear(config.fc4_in_features, config.fc3_in_features)
        self.fc3 = nn.Linear(config.fc3_in_features, config.fc2_in_features)
        self.fc2 = nn.Linear(config.fc2_in_features, config.fc1_in_features)
        self.fc1 = nn.Linear(config.fc1_in_features, config.num_classes)
        
        self.relu = nn.ReLU()
        self.h_score = KL()
        
        # Memory parameters
        self.imgmemory = nn.Parameter(torch.zeros(1, 50))
        self.textmemory = nn.Parameter(torch.zeros(1, 50))
        
        self.init_weight()

    def init_weight(self):
        init.xavier_normal_(self.fc1.weight)
        init.xavier_normal_(self.fc2.weight)
        init.xavier_normal_(self.fc3.weight)
        init.xavier_normal_(self.fc4.weight)
        init.xavier_normal_(self.fc5.weight)
        init.xavier_normal_(self.text_projection.weight)
        init.xavier_normal_(self.image_projection.weight)

    def forward(self, input_ids, attention_mask, image_features):
        bsz = input_ids.size(0)
        
        # Text encoding with BERT
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # Use [CLS] token and all tokens for CNN
        cls_output = bert_output.last_hidden_state[:, 0, :]  # [batch, 768]
        sequence_output = bert_output.last_hidden_state  # [batch, seq_len, 768]
        
        # Project to 300 dimensions for CNN
        text_projected = self.text_projection(sequence_output)  # [batch, seq_len, 300]
        text_projected = text_projected.permute(0, 2, 1)  # [batch, 300, seq_len]
        
        # Text CNN
        conv_block = []
        for Conv in self.convs:
            act = self.relu(Conv(text_projected))
            pool = F.adaptive_max_pool1d(act, 1)
            pool = torch.squeeze(pool, dim=2)
            conv_block.append(pool)
        
        text_cnn_feature = torch.cat(conv_block, dim=1)  # [batch, 400]
        
        # Add dummy features to match expected size
        text_feature_padding = torch.zeros(bsz, 412).to(self.device)
        text_feature = torch.cat([text_cnn_feature, text_feature_padding], dim=1)  # [batch, 812]
        
        # Image features
        image_proj = self.image_projection(image_features)  # [batch, 300]
        image_feature_padding = torch.zeros(bsz, 512).to(self.device)
        image_feature = torch.cat([image_proj, image_feature_padding], dim=1)  # [batch, 812]
        
        # Add memory
        imgmemory = self.imgmemory.expand(bsz, 50)
        image_feature = torch.cat([image_feature, imgmemory], dim=1)  # [batch, 862]
        textmemory = self.textmemory.expand(bsz, 50)
        text_feature = torch.cat([text_feature, textmemory], dim=1)  # [batch, 862]
        
        # Self-attention
        text_feature_view = text_feature.view(bsz, -1, 862)
        image_feature_view = image_feature.view(bsz, -1, 862)
        
        self_att_t = self.dropout(
            self.mh_attention(text_feature_view, text_feature_view, text_feature_view)
        )
        self_att_i = self.dropout(
            self.mh_attention(image_feature_view, image_feature_view, image_feature_view)
        )
        
        self_i = self_att_i.view(bsz, 862)
        self_t = self_att_t.view(bsz, 862)
        
        # Cross-attention
        text_enhanced = self.mh_attention(
            self_att_i.view((bsz, -1, 862)),
            self_att_t.view((bsz, -1, 862)),
            self_att_t.view((bsz, -1, 862))
        ).view(bsz, 862)
        
        self_att_t_final = text_enhanced.view((bsz, -1, 862))
        
        co_att_ti = self.mh_attention(
            self_att_t_final, 
            self_att_i.view(bsz, -1, 862), 
            self_att_i.view(bsz, -1, 862)
        ).view(bsz, 862)
        
        co_att_it = self.mh_attention(
            self_att_i.view(bsz, -1, 862), 
            self_att_t_final, 
            self_att_t_final
        ).view(bsz, 862)
        
        # KL divergence weighting
        skl = self.h_score(self_i, self_t)
        w_unimodel = (1-skl).unsqueeze(1)
        w_mutimodel = skl.unsqueeze(1)
        
        # Fuse features
        att_feature = torch.cat((
            w_unimodel * self_i, 
            w_mutimodel * co_att_it,
            w_mutimodel * co_att_ti, 
            w_unimodel * self_t
        ), dim=1)
        
        # Classification head
        a1 = self.dropout(self.relu(self.bn1(self.fc5(att_feature))))
        a1 = self.dropout(self.relu(self.bn2(self.fc4(a1))))
        a1 = self.dropout(self.relu(self.bn3(self.fc3(a1))))
        a1 = self.dropout_heavy(self.relu(self.bn4(self.fc2(a1))))
        output = self.fc1(a1)
        
        return output

    def fit(self, train_loader, val_loader):
        if torch.cuda.is_available():
            self.cuda()
        
        # Optimizer with different LRs for different parts
        bert_params = list(self.bert.parameters())
        other_params = [p for n, p in self.named_parameters() if 'bert' not in n]
        
        self.optimizer = torch.optim.Adam([
            {'params': bert_params, 'lr': 2e-5},
            {'params': other_params, 'lr': 1e-3}
        ], weight_decay=1e-4)
        
        # Use moderate class weights in loss function
        # For 3.4:1 ratio, use sqrt for balance
        class_weights = torch.tensor([1.0, 1.8]).cuda()
        loss = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.05)
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='max',
            factor=0.5,
            patience=5,
            min_lr=1e-6
        )
        
        patience = 8
        best_val_f1 = 0
        patience_counter = 0
        
        pgd_word = PGD(self, emb_name='bert.embeddings', epsilon=6, alpha=1.8)
        
        print(f"\n{'='*70}")
        print("TRAINING MOPED ON MED-MMHL")
        print(f"{'='*70}\n")
        
        for epoch in range(config.epochs):
            print(f"\n{'='*70}")
            print(f"Epoch {epoch + 1}/{config.epochs}")
            print(f"{'='*70}")
            
            self.train()
            epoch_loss = 0
            epoch_corrects = 0
            epoch_total = 0
            
            epoch_pred_real = 0
            epoch_pred_fake = 0
            epoch_true_real = 0
            epoch_true_fake = 0
            
            pbar = tqdm(train_loader, desc=f"Training", ncols=100)
            
            for i, (batch_input_ids, batch_attention_mask, batch_image_features, batch_labels) in enumerate(pbar):
                batch_input_ids = batch_input_ids.cuda(device=self.device)
                batch_attention_mask = batch_attention_mask.cuda(device=self.device)
                batch_image_features = batch_image_features.cuda(device=self.device)
                batch_labels = batch_labels.cuda(device=self.device)
                
                self.optimizer.zero_grad()
                logits = self.forward(batch_input_ids, batch_attention_mask, batch_image_features)
                loss_classification = loss(logits, batch_labels)
                
                loss_defense = loss_classification
                loss_defense.backward()
                
                # PGD adversarial training after warmup
                if epoch >= 2:
                    K = 3
                    pgd_word.backup_grad()
                    for t in range(K):
                        pgd_word.attack(is_first_attack=(t == 0))
                        if t != K - 1:
                            self.zero_grad()
                        else:
                            pgd_word.restore_grad()
                        loss_adv = self.forward(batch_input_ids, batch_attention_mask, batch_image_features)
                        loss_adv = loss(loss_adv, batch_labels)
                        loss_adv.backward()
                    pgd_word.restore()
                
                self.optimizer.step()
                
                predicted = torch.max(logits, 1)[1]
                corrects = (predicted.view(batch_labels.size()).data == batch_labels.data).sum()
                
                epoch_pred_real += (predicted == 0).sum().item()
                epoch_pred_fake += (predicted == 1).sum().item()
                epoch_true_real += (batch_labels == 0).sum().item()
                epoch_true_fake += (batch_labels == 1).sum().item()
                
                epoch_loss += loss_defense.item()
                epoch_corrects += corrects.item()
                epoch_total += len(batch_labels)
                
                pbar.set_postfix({'loss': f'{loss_defense.item():.4f}'})
            
            avg_loss = epoch_loss / len(train_loader)
            avg_acc = 100 * epoch_corrects / epoch_total
            
            print(f"  Train Loss: {avg_loss:.4f} | Train Acc: {avg_acc:.2f}%")
            print(f"  Ground Truth: Real={epoch_true_real}, Fake={epoch_true_fake}")
            print(f"  Predictions:  Real={epoch_pred_real}, Fake={epoch_pred_fake}")
            
            # Validation
            val_f1 = self.evaluate(val_loader, epoch)
            
            scheduler.step(val_f1)
            
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                patience_counter = 0
                torch.save(self.state_dict(), config.save_path)
                print(f"  Ã¢Å“â€œ Best model saved! (F1: {val_f1:.4f})")
            else:
                patience_counter += 1
                print(f"  No improvement ({patience_counter}/{patience})")
                if patience_counter >= patience:
                    print("Ã¢Å¡Â  Early stopping triggered")
                    break

    def evaluate(self, val_loader, epoch, return_report=False):
        self.eval()
        y_pred = []
        y_true = []
        
        with torch.no_grad():
            for batch_input_ids, batch_attention_mask, batch_image_features, batch_labels in val_loader:
                batch_input_ids = batch_input_ids.cuda(device=self.device)
                batch_attention_mask = batch_attention_mask.cuda(device=self.device)
                batch_image_features = batch_image_features.cuda(device=self.device)
                
                logits = self.forward(batch_input_ids, batch_attention_mask, batch_image_features)
                predicted = torch.max(logits, dim=1)[1]
                
                y_pred.extend(predicted.cpu().numpy().tolist())
                y_true.extend(batch_labels.numpy().tolist())
        
        acc = accuracy_score(y_true, y_pred)
        macro_f1 = f1_score(y_true, y_pred, average="macro")
        report = classification_report(y_true, y_pred, target_names=config.target_names, digits=4)
        
        if macro_f1 > self.best_acc:
            self.best_acc = macro_f1
            
            # Print summary to terminal
            print(f"\n{'-'*70}")
            print(f"Epoch {epoch+1} New Best Macro F1: {macro_f1:.4f}")
            print(report)
            
            # Save report to text file (fold-specific)
            results_file = f"moped_mmhl_best_report_fold{args.fold}.txt"
            with open(results_file, "w") as f:
                f.write("="*20 + f" MoPeD MODEL BEST RESULT (Fold {args.fold}) " + "="*20 + "\n")
                f.write(f"Dataset: Med-MMHL | Epoch: {epoch+1}\n")
                f.write(f"Best Macro F1: {macro_f1:.4f} | Accuracy: {acc:.4f}\n")
                
                cm = confusion_matrix(y_true, y_pred)
                pred_dist = np.bincount(y_pred, minlength=2)
                f.write(f"Predictions: Real={pred_dist[0]}, Fake={pred_dist[1]}\n")
                f.write("-" * 67 + "\n")
                f.write(report)
                f.write("-" * 67 + "\n")
            
            print(f"  Ã¢Å“â€œ Detailed report saved to {results_file}")
        
        y_pred_np = np.array(y_pred)
        y_true_np = np.array(y_true)
        misclassified_idx = np.where(y_pred_np != y_true_np)[0]

        print(f"\n--- Misclassified Samples: {len(misclassified_idx)} ---")
        if len(misclassified_idx) > 0:
            current_df_path = self.config.dev_csv if epoch != 999 else self.config.test_csv
            current_df = pd.read_csv(current_df_path)

            for idx in misclassified_idx[:5]: 
                if idx < len(current_df):
                    print(f"ID: {idx} | Truth: {y_true[idx]} | Pred: {y_pred[idx]}")
                    print(f"Content: {current_df.iloc[idx]['content'][:100]}...")

        if return_report:
            return acc, macro_f1, report
        return macro_f1
    

# ============================================================================
# MAIN TRAINING SCRIPT
# ============================================================================

def main():    
    import warnings    
    warnings.filterwarnings('ignore')
    print("\n" + "="*70)
    print("MoPeD MODEL FOR MED-MMHL DATASET")
    print("="*70 + "\n")
    
    # Initialize tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    # Load datasets
    print("Loading datasets...")
    train_dataset = MedMMHLDataset(
        config.train_csv, 
        config.img_dir, 
        tokenizer, 
        is_training=True
    )
    val_dataset = MedMMHLDataset(
        config.dev_csv,
        config.img_dir, 
        tokenizer, 
        is_training=False
    )
    
    # Analyze class distribution
    train_df = pd.read_csv(config.train_csv)
    test_df = pd.read_csv(config.test_csv)
    
    train_counts = train_df['det_fake_label'].value_counts().sort_index()
    test_counts = test_df['det_fake_label'].value_counts().sort_index()
    
    print(f"\nTRAIN: Real={train_counts[0]}, Fake={train_counts[1]} (Ratio: {train_counts[0]/train_counts[1]:.1f}:1)")
    print(f"TEST:  Real={test_counts[0]}, Fake={test_counts[1]} (Ratio: {test_counts[0]/test_counts[1]:.1f}:1)")
    
    # WARNING: Check for severely imbalanced test set
    test_ratio = test_counts[0] / test_counts[1]
    if test_ratio > 20 or test_counts[1] < 20:
        print(f"\n{'!'*70}")
        print(f"WARNING: Test set has only {test_counts[1]} Fake samples!")
        print(f"This is too few for reliable evaluation. Results will be noisy.")
        print(f"Recommendation: Use stratified split to ensure at least 50+ Fake samples in test.")
        print(f"{'!'*70}\n")
    
    # Weighted sampling - MODERATE upweighting
    # For 3.4:1 ratio (703 Real, 205 Fake), we want to balance but not over-correct
    ratio = train_counts[0] / train_counts[1]
    
    sample_weights = []
    for label in train_df['det_fake_label']:
        if label == 0:  # Real
            sample_weights.append(1.0)
        else:  # Fake - upsample by sqrt of ratio (not full ratio)
            sample_weights.append(np.sqrt(ratio))  # ~1.8x for 3.4:1
    
    print(f"Upsampling minority class by {np.sqrt(ratio):.2f}x")
    
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        sampler=sampler,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True
    )

    # New Test Dataset specifically for this fold
    test_dataset = MedMMHLDataset(
        config.test_csv,
        config.img_dir, 
        tokenizer, 
        is_training=False
    )

    # New Test Loader
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True
    )
    
    # Initialize model
    model = MoPeD_MMHL(config, tokenizer)
    
    # Train
    model.fit(train_loader, val_loader)
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE!")
    print("="*70)
    
    # Final evaluation
    print("\nLoading best model for final evaluation...")
    model.load_state_dict(torch.load(config.save_path))
    
    # CHANGE THIS LINE from val_loader to test_loader
    final_f1 = model.evaluate(test_loader, epoch=999) 
    
    print(f"\nFinal Test Macro F1 for Fold {args.fold}: {final_f1:.4f}")

if __name__ == '__main__':
    main()