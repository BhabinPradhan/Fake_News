import warnings
warnings.filterwarnings('ignore')
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import argparse
import json
from tqdm import tqdm
from PIL import Image

# Essential for metrics and splitting
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split

# Essential for data handling
from collections import defaultdict, Counter

# Model components
import torchvision.models as models
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertModel

# Add argument parser for fold number
parser = argparse.ArgumentParser()
parser.add_argument('--fold', type=int, default=0, help='Fold number (0-4) for k-fold cross validation')
args, _ = parser.parse_known_args()

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    def __init__(self):
        # Adjusted path: XFacta is outside the MoPeD folder
        self.base_dir = "/home/odobasia/Downloads/BERT_Project/XFacta/data"
        self.img_real_dir = os.path.join(self.base_dir, "real_sample/media")
        self.img_fake_dir = os.path.join(self.base_dir, "fake_sample/media")
        
        self.train_json = os.path.join(self.base_dir, "test.json") 
        self.dev_json = os.path.join(self.base_dir, "dev.json")
        
        self.maxlen = 170
        self.num_classes = 2
        self.target_names = ['Real', 'Fake']
        self.batch_size = 16
        self.epochs = 30
        self.save_path = './best_moped_xfacta.pth'
        
        # Kernel sizes for text CNN
        self.kernel_sizes = [2, 3, 4, 5]  # 4 kernels × 100 filters = 400 features
        
        # Feature dimensions (properly calculated, no dummy padding)
        # Text CNN: 4 kernels × 100 filters = 400
        # Image: 300 (from projection)
        # After concatenation with memory: (400+300) + 2×50 = 800
        self.text_cnn_dim = len(self.kernel_sizes) * 100  # 400
        self.image_proj_dim = 300
        self.memory_dim = 50
        
        # Total feature dimension for each modality after adding memory
        self.modal_feature_dim = self.text_cnn_dim + self.image_proj_dim + self.memory_dim  # 750
        
        # Fusion dimension: 4 × modal_feature_dim (self_i, co_att_it, co_att_ti, self_t)
        self.fusion_dim = 4 * self.modal_feature_dim  # 3000
        
        # FC layer dimensions (adjusted for proper feature sizes)
        self.fc5_in_features = self.fusion_dim  # 3000
        self.fc4_in_features = 1024
        self.fc3_in_features = 512
        self.fc2_in_features = 256
        self.fc1_in_features = 128
        self.kl_in_features = 440  # Not used but kept for compatibility

config = Config()

# ============================================================================
# DATASET LOADER
# ============================================================================

class XFactaDataset(Dataset):
    def __init__(self, data_source, config, tokenizer, is_training=False, from_list=False):
        if from_list:
            self.data = data_source
        else:
            with open(data_source, 'r') as f:
                self.data = json.load(f)
        
        self.config = config
        self.tokenizer = tokenizer
        self.is_training = is_training
        
        # FIX: Use functional transforms to avoid NumPy version issues
        self.resize = transforms.Resize((224, 224))
        self.normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        
        self.resnet = models.resnet50(pretrained=True)
        self.resnet.eval()
        for param in self.resnet.parameters():
            param.requires_grad = False
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-1])
        if torch.cuda.is_available():
            self.resnet = self.resnet.cuda()

    def __len__(self):
        return len(self.data)

    def _get_image_path(self, item):
        """Extract image path from XFacta JSON structure"""
        # XFacta provides full paths in the 'images' field
        if 'images' in item and item['images']:
            full_path = item['images'][0]  # Get first image
            
            # The path in JSON is absolute from a different system
            # Example: "/projects/vig/hzy/XFacta/fake_sample/media/batch2/47/images/img0.jpeg"
            # We need to extract the relative part and map to our local path
            
            # Extract the parts we need: batch number and ID
            import re
            match = re.search(r'/(real_sample|fake_sample)/media/batch(\d+)/(\d+)/images/(.+)$', full_path)
            
            if match:
                sample_type = match.group(1)  # real_sample or fake_sample
                batch_num = match.group(2)     # e.g., "2"
                item_id = match.group(3)       # e.g., "47"
                img_name = match.group(4)      # e.g., "img0.jpeg"
                
                # Determine correct root directory
                root = self.config.img_real_dir if sample_type == 'real_sample' else self.config.img_fake_dir
                
                # Construct local path
                local_path = os.path.join(root, f"batch{batch_num}", item_id, "images", img_name)
                
                if os.path.exists(local_path):
                    return local_path
        
        return None

    def __getitem__(self, idx):
        item = self.data[idx]
        img_path = self._get_image_path(item)
        
        image_feature = torch.zeros(2048)
        load_success = False
        
        if img_path and os.path.exists(img_path):
            try:
                img_pil = Image.open(img_path).convert('RGB')
                
                # Resize
                img_resized = self.resize(img_pil)
                
                # Convert PIL to tensor WITHOUT using numpy
                # PIL image is HWC, we need CHW
                from torchvision.transforms import functional as F
                img_tensor = F.pil_to_tensor(img_resized).float() / 255.0
                
                # Apply normalization
                img_tensor = self.normalize(img_tensor).unsqueeze(0)
                
                if torch.cuda.is_available():
                    img_tensor = img_tensor.cuda()
                
                with torch.no_grad():
                    features = self.resnet(img_tensor)
                    image_feature = features.squeeze().cpu()
                    load_success = True
            except Exception as e:
                # Log the error for the first few failures
                if not hasattr(self, '_error_count'):
                    self._error_count = 0
                if self._error_count < 3:
                    print(f"\n⚠️ Image load error #{self._error_count + 1}:")
                    print(f"   Path: {img_path}")
                    print(f"   Error: {str(e)}")
                    import traceback
                    traceback.print_exc()
                    self._error_count += 1

        # Get text from 'text' field (not 'claim')
        text = str(item.get('text', item.get('claim', '')))
        
        # Label is boolean: false=0 (real), true=1 (fake)
        label_value = item.get('label', False)
        label = 1 if label_value else 0
        
        tokens = self.tokenizer(text, padding='max_length', truncation=True, max_length=self.config.maxlen, return_tensors='pt')
        return tokens['input_ids'].squeeze(0), tokens['attention_mask'].squeeze(0), image_feature, label

def collate_fn(batch):
    return (torch.stack([item[0] for item in batch]),
            torch.stack([item[1] for item in batch]),
            torch.stack([item[2] for item in batch]),
            torch.tensor([item[3] for item in batch]))

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
        nn.init.xavier_normal_(self.W_q)
        nn.init.xavier_normal_(self.W_k)
        nn.init.xavier_normal_(self.W_v)
        nn.init.xavier_normal_(self.W_o)

        nn.init.xavier_normal_(self.linear1.weight)
        nn.init.xavier_normal_(self.linear2.weight)

    def FFN(self, X):
        return self.linear2(F.relu(self.linear1(X)))

    def scaled_dot_product_attention(self, Q, K, V):
        batch_size = Q.size(0)
        k_length = K.size(-2)
        
        Q = Q.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, -1, self.n_heads, self.d_v).transpose(1, 2)

        attention_scores = torch.matmul(Q, K.transpose(-1, -2))
        attention_scores = attention_scores / np.sqrt(self.d_k)

        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)

        context = torch.matmul(attention_probs, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.n_heads * self.d_v)
        
        return context

    def forward(self, Q, K, V):
        batch_size = Q.size(0)

        Q_l = torch.matmul(Q, self.W_q)
        K_l = torch.matmul(K, self.W_k)
        V_l = torch.matmul(V, self.W_v)

        context = self.scaled_dot_product_attention(Q_l, K_l, V_l)
        context = torch.matmul(context, self.W_o)

        if self.is_layer_norm:
            context = self.layer_morm(context)
        else:
            context = self.FFN(context)

        return context

class KL(nn.Module):
    def __init__(self):
        super(KL, self).__init__()

    def forward(self, t, v):
        bsz = t.size(0)
        t = t.view(bsz, -1)
        v = v.view(bsz, -1)

        t = F.softmax(t, dim=-1)
        v = F.softmax(v, dim=-1)

        eplison = 1e-7
        kl = (t * ((t + eplison) / (v + eplison)).log()).sum(dim=-1)
        h_score = 1.0 / (1.0 + kl)

        return h_score

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
        self.bert_dropout = nn.Dropout(0.2)  # Reduced from 0.3
        
        # Image feature projection (2048 -> 300)
        self.image_projection = nn.Linear(2048, 300)
        self.image_dropout = nn.Dropout(0.2)  # Reduced from 0.3
        
        # Text CNN layers
        self.dropout = nn.Dropout(0.3)  # Reduced from 0.5
        self.dropout_heavy = nn.Dropout(0.5)  # Reduced from 0.7
        
        self.convs = nn.ModuleList([
            nn.Conv1d(300, 100, kernel_size=K) for K in config.kernel_sizes
        ])
        
        # Attention (using proper modal_feature_dim)
        self.mh_attention = TransformerBlock(
            input_size=config.modal_feature_dim,  # 750
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
        self.imgmemory = nn.Parameter(torch.zeros(1, config.memory_dim))
        self.textmemory = nn.Parameter(torch.zeros(1, config.memory_dim))
        
        self.init_weight()

    def init_weight(self):
        nn.init.xavier_normal_(self.fc1.weight)
        nn.init.xavier_normal_(self.fc2.weight)
        nn.init.xavier_normal_(self.fc3.weight)
        nn.init.xavier_normal_(self.fc4.weight)
        nn.init.xavier_normal_(self.fc5.weight)
        nn.init.xavier_normal_(self.text_projection.weight)
        nn.init.xavier_normal_(self.image_projection.weight)

    def forward(self, input_ids, attention_mask, image_features):
        bsz = input_ids.size(0)
        
        # Text encoding with BERT
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = bert_output.last_hidden_state  # [batch, seq_len, 768]
        
        # Project to 300 dimensions for CNN with dropout
        text_projected = self.bert_dropout(self.text_projection(sequence_output))  # [batch, seq_len, 300]
        text_projected = text_projected.permute(0, 2, 1)  # [batch, 300, seq_len]
        
        # Text CNN - extracts features from different n-gram sizes
        conv_block = []
        for Conv in self.convs:
            act = self.relu(Conv(text_projected))  # [batch, 100, seq_len - k + 1]
            pool = F.adaptive_max_pool1d(act, 1)   # [batch, 100, 1]
            pool = torch.squeeze(pool, dim=2)       # [batch, 100]
            conv_block.append(pool)
        
        text_cnn_feature = torch.cat(conv_block, dim=1)  # [batch, 400]
        text_cnn_feature = self.dropout(text_cnn_feature)  # Dropout on CNN features
        
        # Image features with dropout
        image_proj = self.image_dropout(self.image_projection(image_features))  # [batch, 300]
        
        # Concatenate CNN features with image projection
        text_feature = torch.cat([text_cnn_feature, image_proj], dim=1)  # [batch, 700]
        image_feature = torch.cat([image_proj, text_cnn_feature], dim=1)  # [batch, 700]
        
        # Add memory
        imgmemory = self.imgmemory.expand(bsz, self.config.memory_dim)
        image_feature = torch.cat([image_feature, imgmemory], dim=1)  # [batch, 750]
        
        textmemory = self.textmemory.expand(bsz, self.config.memory_dim)
        text_feature = torch.cat([text_feature, textmemory], dim=1)  # [batch, 750]
        
        # Reshape for attention
        text_feature_view = text_feature.view(bsz, -1, self.config.modal_feature_dim)
        image_feature_view = image_feature.view(bsz, -1, self.config.modal_feature_dim)
        
        # Self-attention
        self_att_t = self.dropout(
            self.mh_attention(text_feature_view, text_feature_view, text_feature_view)
        )
        self_att_i = self.dropout(
            self.mh_attention(image_feature_view, image_feature_view, image_feature_view)
        )
        
        self_i = self_att_i.view(bsz, self.config.modal_feature_dim)
        self_t = self_att_t.view(bsz, self.config.modal_feature_dim)
        
        # Cross-attention
        text_enhanced = self.mh_attention(
            self_att_i.view((bsz, -1, self.config.modal_feature_dim)),
            self_att_t.view((bsz, -1, self.config.modal_feature_dim)),
            self_att_t.view((bsz, -1, self.config.modal_feature_dim))
        ).view(bsz, self.config.modal_feature_dim)
        
        self_att_t_final = text_enhanced.view((bsz, -1, self.config.modal_feature_dim))
        
        co_att_ti = self.mh_attention(
            self_att_t_final, 
            self_att_i.view(bsz, -1, self.config.modal_feature_dim), 
            self_att_i.view(bsz, -1, self.config.modal_feature_dim)
        ).view(bsz, self.config.modal_feature_dim)
        
        co_att_it = self.mh_attention(
            self_att_i.view(bsz, -1, self.config.modal_feature_dim), 
            self_att_t_final, 
            self_att_t_final
        ).view(bsz, self.config.modal_feature_dim)
        
        # KL divergence weighting
        skl = self.h_score(self_i, self_t)
        w_unimodel = (1-skl).unsqueeze(1)
        w_mutimodel = skl.unsqueeze(1)
        
        # Fuse features with KL weighting
        att_feature = torch.cat((
            w_unimodel * self_i, 
            w_mutimodel * co_att_it,
            w_mutimodel * co_att_ti, 
            w_unimodel * self_t
        ), dim=1)  # [batch, 3000]
        
        # Classification head with stronger regularization
        a1 = self.dropout(self.relu(self.bn1(self.fc5(att_feature))))
        a1 = self.dropout(self.relu(self.bn2(self.fc4(a1))))
        a1 = self.dropout(self.relu(self.bn3(self.fc3(a1))))
        a1 = self.dropout_heavy(self.relu(self.bn4(self.fc2(a1))))
        output = self.fc1(a1)
        
        return output

    def evaluate(self, val_loader, epoch):
        self.eval()
        all_preds, all_labels = [], []
        
        with torch.no_grad():
            for ids, mask, img, labels in val_loader:
                if torch.cuda.is_available():
                    ids, mask, img, labels = ids.cuda(), mask.cuda(), img.cuda(), labels.cuda()
                
                logits = self.forward(ids, mask, img)
                preds = torch.argmax(logits, dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        acc = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='weighted')
        
        # Show class distribution of predictions
        from collections import Counter
        pred_dist = Counter(all_preds)
        print(f"\nEpoch {epoch+1} - Val Acc: {acc:.4f}, F1: {f1:.4f}")
        print(f"Prediction distribution: Real={pred_dist[0]}, Fake={pred_dist[1]}")
        print(classification_report(all_labels, all_preds, target_names=config.target_names, digits=4, zero_division=0))
        
        if acc > self.best_acc:
            self.best_acc = acc
            torch.save(self.state_dict(), config.save_path)
            print(f"✓ Model saved with accuracy: {acc:.4f}")
        
        return acc, f1  # Return both for scheduler and early stopping

    def fit(self, train_loader, val_loader):
        if torch.cuda.is_available(): 
            self.cuda()
        
        # CRITICAL: Balanced class weighting
        # Train has: {1: 960, 0: 959} - nearly balanced
        # Start with equal weights and let the model learn naturally
        weights = torch.tensor([1.5, 1.0])  # Moderate favor to Real to balance the dataset
        if torch.cuda.is_available():
            weights = weights.cuda()
        loss_fn = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.1)
        
        # Improved optimizer - balanced learning rates
        self.optimizer = torch.optim.AdamW([
            {'params': list(self.bert.parameters()), 'lr': 2e-5},
            {'params': [p for n, p in self.named_parameters() if 'bert' not in n], 'lr': 3e-4}
        ], weight_decay=1e-3)  # Moderate weight decay
        
        # Learning rate scheduler - moderate patience
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', factor=0.5, patience=3
        )
        
        # Add early stopping patience
        patience_counter = 0
        best_f1 = 0

        for epoch in range(config.epochs):
            self.train()
            total_loss = 0
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
            
            for ids, mask, img, labels in pbar:
                if torch.cuda.is_available():
                    ids, mask, img, labels = ids.cuda(), mask.cuda(), img.cuda(), labels.cuda()
                
                self.optimizer.zero_grad()
                logits = self.forward(ids, mask, img)
                l = loss_fn(logits, labels)
                l.backward()
                
                # Gradient clipping to prevent instability
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                
                self.optimizer.step()
                total_loss += l.item()
                pbar.set_postfix({'loss': f'{l.item():.4f}'})
            
            avg_loss = total_loss / len(train_loader)
            print(f"Epoch {epoch+1} - Avg Loss: {avg_loss:.4f}")
            
            # Evaluate and adjust learning rate
            val_acc, val_f1 = self.evaluate(val_loader, epoch)
            scheduler.step(val_acc)
            
            # Early stopping based on F1
            if val_f1 > best_f1:
                best_f1 = val_f1
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= 5:
                    print(f"\n⚠️ Early stopping triggered after {epoch+1} epochs")
                    break

# ============================================================================
# MAIN
# ============================================================================

def main():
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    with open(config.train_json, 'r') as f: d1 = json.load(f)
    with open(config.dev_json, 'r') as f: d2 = json.load(f)
    all_samples = d1 + d2

    # DEBUG: Print first sample to see actual JSON structure
    print("\n🔍 First sample from JSON:")
    print(json.dumps(all_samples[0], indent=2))
    
    # Image-based Split to prevent leakage
    # Extract batch and ID from the image path in JSON
    image_to_samples = defaultdict(list)
    for idx, sample in enumerate(all_samples):
        if 'images' in sample and sample['images']:
            img_path = sample['images'][0]
            # Extract batch and ID from path like: "/projects/.../batch2/47/images/img0.jpeg"
            import re
            match = re.search(r'/batch(\d+)/(\d+)/', img_path)
            if match:
                batch = match.group(1)
                item_id = match.group(2)
                image_to_samples[f"{batch}_{item_id}"].append(idx)
            else:
                # Fallback to using index
                image_to_samples[f"idx_{idx}"].append(idx)
        else:
            image_to_samples[f"idx_{idx}"].append(idx)
    
    u_imgs = list(image_to_samples.keys())
    tr_imgs, te_imgs = train_test_split(u_imgs, test_size=0.2, random_state=42)
    
    tr_data = [all_samples[i] for img in tr_imgs for i in image_to_samples[img]]
    te_data = [all_samples[i] for img in te_imgs for i in image_to_samples[img]]

    print(f"Train Dist: {Counter([1 if s.get('label') else 0 for s in tr_data])}")
    
    train_ds = XFactaDataset(tr_data, config, tokenizer, is_training=True, from_list=True)
    val_ds = XFactaDataset(te_data, config, tokenizer, is_training=False, from_list=True)
    
    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False, collate_fn=collate_fn)
    
    # Audit for Image Loading
    _, _, test_img, _ = next(iter(train_loader))
    zeros = (test_img.sum(dim=1) == 0).sum().item()
    print(f"Audit: {zeros}/{config.batch_size} images failed to load. Check paths if high.")
    
    # Manual test: Try loading one image directly
    print("\n🧪 Manual image load test:")
    test_sample = tr_data[0]
    test_dataset = train_ds
    test_img_path = test_dataset._get_image_path(test_sample)
    print(f"  Path exists: {os.path.exists(test_img_path) if test_img_path else False}")
    print(f"  Path: {test_img_path}")
    
    if test_img_path and os.path.exists(test_img_path):
        try:
            from PIL import Image
            from torchvision.transforms import functional as F
            img = Image.open(test_img_path)
            print(f"  ✓ PIL can open: {img.size}, mode={img.mode}")
            img_rgb = img.convert('RGB')
            print(f"  ✓ Converted to RGB: {img_rgb.size}")
            
            # Use pure PyTorch approach
            resize = transforms.Resize((224, 224))
            normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            
            img_resized = resize(img_rgb)
            print(f"  ✓ Resized: {img_resized.size}")
            
            # Use torchvision functional API (no NumPy)
            img_tensor = F.pil_to_tensor(img_resized).float() / 255.0
            print(f"  ✓ Converted to tensor: {img_tensor.shape}")
            
            img_tensor = normalize(img_tensor)
            print(f"  ✓ Normalized: {img_tensor.shape}")
            
            # Try ResNet
            resnet = models.resnet50(pretrained=True)
            resnet.eval()
            resnet = nn.Sequential(*list(resnet.children())[:-1])
            if torch.cuda.is_available():
                resnet = resnet.cuda()
                img_tensor = img_tensor.unsqueeze(0).cuda()
            else:
                img_tensor = img_tensor.unsqueeze(0)
            
            with torch.no_grad():
                features = resnet(img_tensor)
                print(f"  ✓ ResNet extraction works: {features.shape}")
                print(f"  ✓ Feature sum (should be non-zero): {features.sum().item():.2f}")
        except Exception as e:
            print(f"  ✗ Error during manual test: {e}")
            import traceback
            traceback.print_exc()
    
    
    # FIX 3: Diagnostic for image paths
    if zeros > 0:
        print("\n⚠ Image Loading Diagnostics:")
        print(f"Real images directory exists: {os.path.exists(config.img_real_dir)}")
        print(f"Fake images directory exists: {os.path.exists(config.img_fake_dir)}")
        
        # Show a sample of what the actual directory structure looks like
        if os.path.exists(config.img_real_dir):
            print(f"\nSample structure in real_dir:")
            for item in os.listdir(config.img_real_dir)[:3]:
                item_path = os.path.join(config.img_real_dir, item)
                if os.path.isdir(item_path):
                    print(f"  {item}/")
                    subitems = os.listdir(item_path)[:3]
                    for subitem in subitems:
                        subitem_path = os.path.join(item_path, subitem)
                        print(f"    {subitem}/")
                        # Show what's inside the ID folder
                        if os.path.isdir(subitem_path):
                            contents = os.listdir(subitem_path)[:5]
                            for content in contents:
                                content_path = os.path.join(subitem_path, content)
                                if os.path.isdir(content_path):
                                    print(f"      📁 {content}/")
                                    # Show images inside
                                    imgs = [f for f in os.listdir(content_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))][:3]
                                    for img in imgs:
                                        print(f"        🖼️  {img}")
                                else:
                                    print(f"      📄 {content}")
        
        # Test with a real sample from the training data
        print("\n🔍 Testing path resolution with first training sample:")
        sample = tr_data[0]
        
        if 'images' in sample and sample['images']:
            img_path_from_json = sample['images'][0]
            print(f"  Image path in JSON: {img_path_from_json}")
            
            # Try to extract and build local path
            import re
            match = re.search(r'/(real_sample|fake_sample)/media/batch(\d+)/(\d+)/images/(.+)$', img_path_from_json)
            if match:
                sample_type = match.group(1)
                batch = match.group(2)
                sid = match.group(3)
                img_name = match.group(4)
                
                root = config.img_real_dir if sample_type == 'real_sample' else config.img_fake_dir
                local_path = os.path.join(root, f"batch{batch}", sid, "images", img_name)
                
                print(f"  Extracted: type={sample_type}, batch={batch}, id={sid}, img={img_name}")
                print(f"  {'✓' if os.path.exists(local_path) else '✗'} Local path: {local_path}")
            else:
                print(f"  ✗ Could not parse image path from JSON")


    model = MoPeD_MMHL(config, tokenizer)
    model.fit(train_loader, val_loader)

if __name__ == '__main__':
    main()