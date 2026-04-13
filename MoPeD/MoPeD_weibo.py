import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.nn.utils as utils
from sklearn.metrics import classification_report, accuracy_score
import torchvision.models as models
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from sklearn.metrics import f1_score

import torch.nn.init as init
import pickle
import abc
import json, os
import argparse
import config_file
import random
from PIL import Image
import sys
from torch.distributions import Normal, Independent
from torch.nn.functional import softplus
sys.path.append('/../image_part')
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.description = "ini"
parser.add_argument("-t", "--task", type=str, default="weibo")
parser.add_argument("-g", "--gpu_id", type=str, default="1")
parser.add_argument("-c", "--config_name", type=str, default="single3.json")
parser.add_argument("-T", "--thread_name", type=str, default="Thread-1")
parser.add_argument("-d", "--description", type=str, default="exp_description")
# Modify the argument defaults for smaller hidden layers:
parser.add_argument('--fc5_in_features', type=int, default=3448, help='...')
parser.add_argument('--fc4_in_features', type=int, default=1024) 
parser.add_argument('--fc3_in_features', type=int, default=512)  
parser.add_argument('--fc2_in_features', type=int, default=500, help='...')  
parser.add_argument('--fc1_in_features', type=int, default=200, help='...')   
parser.add_argument('--kl_in_features', type=int, default=440, help='Number of input features for kl')
args, _ = parser.parse_known_args()


def process_config(config):
    for k, v in config.items():
        config[k] = v[0] if isinstance(v, list) else v
    return config

# Module-level config so classes can reference config['key'] at import time
config = process_config(config_file.config)

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

class NeuralNetwork(nn.Module):

    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.best_acc = 0
        self.init_clip_max_norm = None
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    @abc.abstractmethod
    def forward(self):
        pass

    # Add 'epoch' to the arguments
    def moped(self, x_tid, x_text, y, loss, i, total, params, pgd_word, image_features, text_features, epoch):
        self.optimizer.zero_grad()
        logit_defense = self.forward(x_tid, x_text, image_features, text_features)
        loss_classification = loss(logit_defense, y)

        loss_defense = loss_classification
        loss_defense.backward()

        # Wrap PGD logic in a warm-up condition
        if epoch > 2: # Only start adversarial training after the model is stable (Epoch 3+)
            K = 3
            pgd_word.backup_grad()
            for t in range(K):
                pgd_word.attack(is_first_attack=(t == 0))
                if t != K - 1:
                    self.zero_grad()
                else:
                    pgd_word.restore_grad()
                loss_adv = self.forward(x_tid, x_text, image_features, text_features)
                loss_adv = loss(loss_adv, y)
                loss_adv.backward()
            pgd_word.restore()

        self.optimizer.step()

    def fit(self, X_train_tid, X_train, y_train, X_dev_tid, X_dev, y_dev,
        image_features1, text_features1, image_features2, text_features2):

        if torch.cuda.is_available():
            self.cuda()
        
        batch_size = self.config['batch_size']
        
        # MODERATE LEARNING RATE
        self.optimizer = torch.optim.Adam(self.parameters(), lr=8e-4, weight_decay=8e-5)

        X_train_tid = torch.LongTensor(X_train_tid)
        X_train = torch.LongTensor(X_train)
        y_train = torch.LongTensor(y_train)

        # OPTION 1: Use weighted sampling WITHOUT class weights in loss
        # This is gentler than double-penalization
        from torch.utils.data import WeightedRandomSampler

        class_sample_counts = np.bincount(y_train.numpy())

        # Inverse frequency
        class_weights = 1.0 / class_sample_counts

        # ?? CLIP to avoid extreme oversampling
        class_weights = np.clip(class_weights, 0.8, 1.2)

        # Map class weights to samples
        sample_weights = class_weights[y_train.numpy()]

        sampler = WeightedRandomSampler(
            weights=torch.DoubleTensor(sample_weights),
            num_samples=len(sample_weights),
            replacement=True
        )
        
        print(f"\n{'='*70}")
        print("TRAINING CONFIGURATION")
        print(f"{'='*70}")
        print(f"Training samples: {len(y_train)}")
        unique, counts = np.unique(y_train.numpy(), return_counts=True)
        for cls, count in zip(unique, counts):
            class_name = self.config['target_names'][cls]
            print(f"  Class {cls} ({class_name}): {count:4d} samples ({100*count/len(y_train):5.1f}%)")
        print(f"Using weighted sampling (no class weights in loss)")
        print(f"Batch size: {batch_size}")
        print(f"Learning rate: {self.optimizer.param_groups[0]['lr']}")
        print(f"Weight decay: {self.optimizer.param_groups[0]['weight_decay']}")
        print(f"Total epochs: {self.config['epochs']}")
        print(f"{'='*70}\n")

        dataset = TensorDataset(X_train_tid, X_train, y_train, image_features1, text_features1)
        dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)
        
        loss = nn.CrossEntropyLoss(label_smoothing=0.05)
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 
            mode='max',
            factor=0.5,
            patience=5,
            min_lr=1e-6
        )
        
        patience = 8  # Increase patience
        best_val_acc = 0
        patience_counter = 0
        
        params = [(name, param) for name, param in self.named_parameters()]
        pgd_word = PGD(self, emb_name='word_embedding', epsilon=6, alpha=1.8)
        
        for epoch in range(self.config['epochs']):
            print(f"\n{'='*70}")
            print(f"Epoch {epoch + 1}/{self.config['epochs']}")
            print(f"{'='*70}")
            
            self.train()
            epoch_loss = 0
            epoch_corrects = 0
            epoch_total = 0
            
            # Track predictions per class
            epoch_pred_nr = 0
            epoch_pred_fr = 0
            epoch_true_nr = 0
            epoch_true_fr = 0
            
            pbar = tqdm(dataloader, desc=f"Training", ncols=100, 
                    bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')
            
            for i, data in enumerate(pbar):
                batch_x_tid, batch_x_text, batch_y, batch_image_features, batch_text_features = (
                    item.cuda(device=self.device) for item in data
                )
                
                self.optimizer.zero_grad()
                logit_defense = self.forward(batch_x_tid, batch_x_text, batch_image_features, batch_text_features)
                loss_classification = loss(logit_defense, batch_y)

                loss_defense = loss_classification
                loss_defense.backward()
                
                # ===== PGD only AFTER warm-up =====
                if epoch >= 2:
                    K = 3
                    pgd_word.backup_grad()
                    for t in range(K):
                        pgd_word.attack(is_first_attack=(t == 0))
                        if t != K - 1:
                            self.zero_grad()
                        else:
                            pgd_word.restore_grad()
                        loss_adv = self.forward(
                            batch_x_tid, batch_x_text,
                            batch_image_features, batch_text_features
                        )
                        loss_adv = loss(loss_adv, batch_y)
                        loss_adv.backward()
                    pgd_word.restore()

                self.optimizer.step()
                
                predicted = torch.max(logit_defense, 1)[1]
                corrects = (predicted.view(batch_y.size()).data == batch_y.data).sum()
                
                # Count predictions and ground truth
                epoch_pred_nr += (predicted == 0).sum().item()
                epoch_pred_fr += (predicted == 1).sum().item()
                epoch_true_nr += (batch_y == 0).sum().item()
                epoch_true_fr += (batch_y == 1).sum().item()
                
                epoch_loss += loss_defense.item()
                epoch_corrects += corrects.item()
                epoch_total += len(batch_y)

                if self.init_clip_max_norm is not None:
                    utils.clip_grad_norm_(self.parameters(), max_norm=self.init_clip_max_norm)
            
            avg_loss = epoch_loss / len(dataloader)
            avg_acc = 100 * epoch_corrects / epoch_total
            
            # Show detailed statistics
            print(f"  Train Loss: {avg_loss:.4f} | Train Acc: {avg_acc:.2f}% ({epoch_corrects}/{epoch_total})")
            print(f"  Ground Truth: NR={epoch_true_nr} ({100*epoch_true_nr/epoch_total:.1f}%), FR={epoch_true_fr} ({100*epoch_true_fr/epoch_total:.1f}%)")
            print(f"  Predictions:  NR={epoch_pred_nr} ({100*epoch_pred_nr/epoch_total:.1f}%), FR={epoch_pred_fr} ({100*epoch_pred_fr/epoch_total:.1f}%)")
            
            # Warning if severely imbalanced predictions
            nr_pred_ratio = epoch_pred_nr / epoch_total
            if nr_pred_ratio < 0.10:
                print(f"  ? WARNING: Model barely predicting NR class ({nr_pred_ratio*100:.1f}%)")
            elif nr_pred_ratio > 0.90:
                print(f"  ? WARNING: Model over-predicting NR class ({nr_pred_ratio*100:.1f}%)")
            
            val_acc = self.evaluate(
                X_dev_tid, X_dev, y_dev,
                image_features2, text_features2,
                epoch
            )

            
            scheduler.step(val_acc)
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print("?? Early stopping triggered")
                    break

            current_lr = self.optimizer.param_groups[0]['lr']
            print(f"  Learning Rate: {current_lr:.6f}")

    def evaluate(self, X_dev_tid, X_dev, y_dev,
             image_features2, text_features2,
             epoch):
        y_pred = self.predict(X_dev_tid, X_dev, image_features2, text_features2)

        macro_f1 = f1_score(y_dev, y_pred, average="macro")

        if epoch >= 3 and macro_f1 > self.best_acc:
            self.best_acc = macro_f1
            torch.save(self.state_dict(), self.config['save_path'])

            from sklearn.metrics import confusion_matrix
            print(f"\n{'-'*70}")
            print(classification_report(
                y_dev, y_pred,
                target_names=self.config['target_names'],
                digits=5
            ))

            unique_preds = set(y_pred)
            if len(unique_preds) < 2:
                print("? Skipping save: only one class predicted")
                return macro_f1

            cm = confusion_matrix(y_dev, y_pred)
            print(f"\nConfusion Matrix:")
            print(f"              Predicted")
            print(f"              NR    FR")
            print(f"Actual NR   {cm[0][0]:4d}  {cm[0][1]:4d}")
            print(f"       FR   {cm[1][0]:4d}  {cm[1][1]:4d}")

            tn, fp, fn, tp = cm.ravel()
            print(f"\nDetailed Metrics:")
            print(f"  TN (NR?NR): {tn:4d} | FP (NR?FR): {fp:4d}")
            print(f"  FN (FR?NR): {fn:4d} | TP (FR?FR): {tp:4d}")
            print(f"{'-'*70}")

            print(f"Val Macro-F1: {macro_f1:.4f} | Saved: {self.config['save_path']}")

        return macro_f1


    def predict(self, X_test_tid, X_test, image_features3, text_features3):
        if torch.cuda.is_available():
            self.cuda()
        self.eval() 
        y_pred = []
        X_test_tid = torch.LongTensor(X_test_tid).cuda()
        X_test = torch.LongTensor(X_test).cuda()

        dataset = TensorDataset(X_test_tid, X_test, image_features3, text_features3)
        dataloader = DataLoader(dataset, batch_size=50)

        for i, data in enumerate(dataloader):
            with torch.no_grad():
                batch_x_tid, batch_x_text, batch_image_features, batch_text_features = (
                    item.cuda(device=self.device) for item in data
                )
                logits = self.forward(batch_x_tid, batch_x_text, batch_image_features, batch_text_features)
                predicted = torch.max(logits, dim=1)[1]
                y_pred += predicted.data.cpu().numpy().tolist()
        return y_pred

class resnet50(nn.Module):
    def __init__(self):
        super(resnet50, self).__init__()
        self.X_img = config.get('X_img', [])
        self.fc = nn.Linear(2048, 300)
        torch.nn.init.eye_(self.fc.weight)

    def forward(self, X_tid):
        # During inference self.X_img is empty — return zeros so downstream concat works.
        bsz = X_tid.shape[0]
        if len(self.X_img) == 0:
            return torch.zeros(bsz, 300).to(X_tid.device)

        img_list = []
        for newid in X_tid.cpu().numpy():
            safe_id = int(newid)
            if safe_id < 0 or safe_id >= len(self.X_img):
                safe_id = 0
            img_feature = self.X_img[safe_id]
            if isinstance(img_feature, np.ndarray):
                img_tensor = torch.from_numpy(img_feature).float()
            else:
                img_tensor = torch.tensor(img_feature).float()
            img_list.append(img_tensor.unsqueeze(0))

        batch_img = torch.cat(img_list, dim=0).to(X_tid.device)
        img_output = self.fc(batch_img)
        return img_output

class Encoder(nn.Module):
    def __init__(self, z_dim=2):
        super(Encoder, self).__init__()
        self.z_dim = z_dim
        self.net = nn.Sequential(
            nn.Linear(862, args.kl_in_features),
            nn.ReLU(True),
            nn.Linear(args.kl_in_features, z_dim * 2),
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

class Config:
    """Importable config wrapper so moped_wrapper.py can do: from MoPeD_weibo import MoPeD, Config"""
    def __init__(self, overrides=None):
        import config_file as _cf
        self._data = process_config(dict(_cf.config))
        if overrides:
            self._data.update(overrides)
    def __getitem__(self, key): return self._data[key]
    def __setitem__(self, key, value): self._data[key] = value
    def __contains__(self, key): return key in self._data
    def get(self, key, default=None): return self._data.get(key, default)
    def update(self, d): self._data.update(d)
    def items(self): return self._data.items()
    def keys(self): return self._data.keys()


class MoPeD(NeuralNetwork):
    def __init__(self, config):
        super(MoPeD, self).__init__()
        self.config = config
        embedding_weights = config['embedding_weights']
        
        if isinstance(embedding_weights, np.ndarray):
            weight_tensor = torch.from_numpy(embedding_weights)
        else:
            weight_tensor = embedding_weights
            
        V, D = weight_tensor.shape
        maxlen = config['maxlen']
        
        self.dropout = nn.Dropout(0.6)
        self.dropout_heavy = nn.Dropout(0.8)
        
        self.mh_attention = TransformerBlock(input_size=862, n_heads=8, attn_dropout=0.5)

        self.word_embedding = nn.Embedding(
            num_embeddings=V, 
            embedding_dim=D, 
            padding_idx=0,
            _weight=weight_tensor.float()
        )
        
        self.image_embedding = resnet50()
        self.convs = nn.ModuleList([nn.Conv1d(300, 100, kernel_size=K) for K in config['kernel_sizes']])
        self.max_poolings = nn.ModuleList([nn.MaxPool1d(kernel_size=maxlen - K + 1) for K in config['kernel_sizes']])

        self.relu = nn.ReLU()
        
        # Modify batch norm layers for better regularization:
        self.bn1 = nn.BatchNorm1d(args.fc4_in_features, momentum=0.05)  
        self.bn2 = nn.BatchNorm1d(args.fc3_in_features, momentum=0.05)
        self.bn3 = nn.BatchNorm1d(args.fc2_in_features, momentum=0.05)
        self.bn4 = nn.BatchNorm1d(args.fc1_in_features, momentum=0.05)
        
        self.fc5 = nn.Linear(args.fc5_in_features, args.fc4_in_features)
        self.fc4 = nn.Linear(args.fc4_in_features, args.fc3_in_features)
        self.fc3 = nn.Linear(args.fc3_in_features, args.fc2_in_features)
        self.fc2 = nn.Linear(args.fc2_in_features, args.fc1_in_features)
        self.fc1 = nn.Linear(in_features=args.fc1_in_features, out_features=config['num_classes'])
        self.init_weight()

        self.h_score = KL()
        self.imgmemory = nn.Parameter(torch.zeros(1, 50))
        self.textmemory = nn.Parameter(torch.zeros(1, 50))

    def init_weight(self):
        init.xavier_normal_(self.fc1.weight)
        init.xavier_normal_(self.fc2.weight)
        init.xavier_normal_(self.fc3.weight)
        init.xavier_normal_(self.fc4.weight)
        init.xavier_normal_(self.fc5.weight)

    def forward(self, X_tid, X_text, image_features, text_features):
        # Only remove extra leading dimensions, never collapse below 2D
        while image_features.dim() > 2:
            image_features = image_features.squeeze(1)
        while text_features.dim() > 2:
            text_features = text_features.squeeze(1)
        
        # Clamp token IDs to valid vocab range — prevents CUDA out-of-bounds
        # when BERT tokenizer IDs exceed the model's custom vocabulary size
        X_text = X_text.clamp(0, self.word_embedding.num_embeddings - 1)
        X_text = self.word_embedding(X_text)

        # 1. Text Self-Attention with Dropout
        if self.config['user_self_attention'] == True:
            X_text = self.dropout(self.mh_attention(X_text, X_text, X_text))
            
        X_text = X_text.permute(0, 2, 1)
        iembedding = self.image_embedding.forward(X_tid)
        image_feature = torch.cat([iembedding, image_features], dim=1)
        
        conv_block = []
        for _, (Conv, max_pooling) in enumerate(zip(self.convs, self.max_poolings)):
            act = self.relu(Conv(X_text))
            pool = max_pooling(act)
            # Keep batch dimension when bsz==1; output must stay [bsz, channels]
            pool = pool.squeeze(-1)
            conv_block.append(pool)

        text_feature = torch.cat(conv_block, dim=1)
        text_feature = torch.cat([text_feature, text_features], dim=1)
        bsz = text_feature.size()[0]

        imgmemory = self.imgmemory.expand(bsz, 50)
        image_feature = torch.cat([image_feature, imgmemory], dim=1)
        textmemory = self.textmemory.expand(bsz, 50)
        text_feature = torch.cat([text_feature, textmemory], dim=1)

        # 2. Multimodal Self-Attention with Dropout (Corrected from ... placeholder)
        self_att_t = self.dropout(self.mh_attention(text_feature.view(bsz, -1, 862), 
                                                   text_feature.view(bsz, -1, 862), 
                                                   text_feature.view(bsz, -1, 862)))

        self_att_i = self.dropout(self.mh_attention(image_feature.view(bsz, -1, 862), 
                                                   image_feature.view(bsz, -1, 862), 
                                                   image_feature.view(bsz, -1, 862)))

        self_i = self_att_i.view(bsz, 862)
        self_t = self_att_t.view(bsz, 862)

        # 3. Enhanced Fusion Attention
        text_enhanced = self.mh_attention(self_att_i.view((bsz, -1, 862)), 
                                        self_att_t.view((bsz, -1, 862)), 
                                        self_att_t.view((bsz, -1, 862))).view(bsz, 862)

        self_att_t_final = text_enhanced.view((bsz, -1, 862))

        co_att_ti = self.mh_attention(self_att_t_final, self_att_i.view(bsz, -1, 862), self_att_i.view(bsz, -1, 862)).view(bsz, 862)
        co_att_it = self.mh_attention(self_att_i.view(bsz, -1, 862), self_att_t_final, self_att_t_final).view(bsz, 862)

        skl = self.h_score(self_i, self_t)
        w_unimodel = (1-skl).unsqueeze(1)
        w_mutimodel = skl.unsqueeze(1)
        
        att_feature = torch.cat((w_unimodel * self_i, w_mutimodel * co_att_it, 
                                w_mutimodel * co_att_ti, w_unimodel * self_t), dim=1)
        
        # 4. Final Classification with Batch Norm and Dropout
        a1 = self.dropout(self.relu(self.bn1(self.fc5(att_feature))))
        a1 = self.dropout(self.relu(self.bn2(self.fc4(a1))))
        a1 = self.dropout(self.relu(self.bn3(self.fc3(a1))))
        a1 = self.dropout_heavy(self.relu(self.bn4(self.fc2(a1))))
        output = self.fc1(a1)

        return output
        
def load_dataset():
    import torch
    import numpy as np
    import pickle
    import os
    
    pre = '../dataset/weibo/weibo_files'

    def process_features(features):
        tensors = [torch.from_numpy(x).float() if isinstance(x, np.ndarray) else torch.tensor(x).float() for x in features]
        return torch.stack(tensors).to('cpu')

    print("\n" + "="*60)
    print("LOADING AND INSPECTING PICKLE FILES")
    print("="*60)
    
    # 1. Load ALL text data first to ensure variables are assigned
    with open(pre + "/train.pkl", 'rb') as f:
        train_data = pickle.load(f)
    with open(pre + "/dev.pkl", 'rb') as f:
        dev_data = pickle.load(f)
    with open(pre + "/test.pkl", 'rb') as f:
        test_data = pickle.load(f)

    # Extract data from pickles
    post_ids_train, X_train, y_train, word_embeddings = train_data
    _, X_dev, y_dev, _ = dev_data
    _, X_test, y_test, _ = test_data

    # 2. Check for embedding corruption
    if word_embeddings.shape[0] == 1:
        print("\n" + "!"*60)
        print("ERROR: Word embeddings has only 1 row! Fixing...")
        print("!"*60)
        
        # Convert to arrays to find max ID safely
        X_train_array = np.array(X_train)
        X_dev_array = np.array(X_dev)
        X_test_array = np.array(X_test)
        
        max_token_id = max(int(np.max(X_train_array)), 
                        int(np.max(X_dev_array)), 
                        int(np.max(X_test_array)))
        
        vocab_size = max_token_id + 1
        print(f"Creating random embeddings for vocab_size={vocab_size}")
        
        embeddings_path = pre + "/word_embeddings.npy"
        if os.path.exists(embeddings_path):
            print(f"Loading pre-trained embeddings from {embeddings_path}")
            word_embeddings = np.load(embeddings_path)
        else:
            word_embeddings = np.random.randn(vocab_size, 300).astype(np.float32) * 0.01

    # 3. Text Data Validation
    vocab_size = word_embeddings.shape[0]
    X_train, X_dev, X_test = np.array(X_train), np.array(X_dev), np.array(X_test)
    
    # Clip any out-of-bounds indices to prevent Embedding layer crashes
    X_train = np.clip(X_train, 0, vocab_size - 1)
    X_dev = np.clip(X_dev, 0, vocab_size - 1)
    X_test = np.clip(X_test, 0, vocab_size - 1)
    
    print(f"Validation Complete: All indices in range [0, {vocab_size-1}]")
    
    # 4. Load CLIP features
    with open(pre + "/Weiboclip_train.pkl", 'rb') as f:
        img_f1, text_f1, _ = pickle.load(f)
    with open(pre + "/Weiboclip_dev.pkl", 'rb') as f:
        img_f2, text_f2, _ = pickle.load(f)
    with open(pre + "/Weiboclip_test.pkl", 'rb') as f:
        img_f3, text_f3, _ = pickle.load(f)

    # 5. Load ResNet features
    with open(pre + "/X_img.pkl", 'rb') as f:
        img_data = pickle.load(f)
    train_res, val_res, test_res = img_data

    # 6. Safety Alignment
    train_len = min(len(X_train), len(img_f1), len(train_res))
    val_len = min(len(X_dev), len(img_f2), len(val_res))
    test_len = min(len(X_test), len(img_f3), len(test_res))

    X_train, y_train = X_train[:train_len], y_train[:train_len]
    img_f1, text_f1 = img_f1[:train_len], text_f1[:train_len]
    train_res = train_res[:train_len]

    X_dev, y_dev = X_dev[:val_len], y_dev[:val_len]
    img_f2, text_f2 = img_f2[:val_len], text_f2[:val_len]
    val_res = val_res[:val_len]

    X_test, y_test = X_test[:test_len], y_test[:test_len]
    img_f3, text_f3 = img_f3[:test_len], text_f3[:test_len]
    test_res = test_res[:test_len]

    # Create Lookup Table
    config['X_img'] = train_res + val_res + test_res
    X_train_tid = np.array(range(0, train_len))
    X_dev_tid   = np.array(range(train_len, train_len + val_len))
    X_test_tid  = np.array(range(train_len + val_len, train_len + val_len + test_len))

    config['embedding_weights'] = word_embeddings

    image_features1, text_features1 = process_features(img_f1), process_features(text_f1)
    image_features2, text_features2 = process_features(img_f2), process_features(text_f2)
    image_features3, text_features3 = process_features(img_f3), process_features(text_f3)

    return X_train_tid, X_train, y_train, \
        X_dev_tid, X_dev, y_dev, \
        X_test_tid, X_test, y_test, \
        image_features1, text_features1, y_train, \
        image_features2, text_features2, y_dev, \
        image_features3, text_features3, y_test

def train_and_test(model):
    save_path = './best_moped_weibo.pth'
    
    if os.path.exists(save_path):
        os.remove(save_path)

    # 1. Load the dataset
    dataset_output = load_dataset()
    
    # Extracting the features for the fit() method
    (X_train_tid, X_train, y_train, 
     X_dev_tid, X_dev, y_dev, 
     X_test_tid, X_test, y_test, 
     image_features1, text_features1, _, 
     image_features2, text_features2, _, 
     image_features3, text_features3, _) = dataset_output

    # 2. Configuration for Weibo 
    config['save_path'] = save_path
    config['batch_size'] = 32
    config['epochs'] = 30
    config['maxlen'] = 170 
    config['kernel_sizes'] = [2, 3, 4] 

    # Critical fixes for the missing keys
    config['user_self_attention'] = False  
    config['num_classes'] = 2
    config['target_names'] = ['Pristine', 'OOC']
    
    # 3. Initialize model with the fully populated config

    model_instance = model(config)
    
    print(f"\nStarting Retraining. Weights will be saved to: {save_path}")
    
    # 4. Run the training loop
    model_instance.fit(X_train_tid, X_train, y_train,
                       X_dev_tid, X_dev, y_dev, 
                       image_features1, text_features1, 
                       image_features2, text_features2)

    # 5. Final Verification
    if os.path.exists(save_path):
        print(f"✓ SUCCESS: Model weights saved to {save_path}")
        model_instance.load_state_dict(torch.load(save_path))
        y_pred = model_instance.predict(X_test_tid, X_test, image_features3, text_features3)
        from sklearn.metrics import classification_report
        print(classification_report(y_test, y_pred, target_names=['Real', 'Fake']))
    else:
        print("✗ ERROR: Model failed to save. Ensure best Macro-F1 was hit after Epoch 3.")

if __name__ == "__main__":
    # Only runs when executing directly: python MoPeD_weibo.py
    # Skipped entirely when imported by moped_wrapper.py or model_manager.py
    seed = config['seed']
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.manual_seed(seed)

    model = MoPeD
    model_result = train_and_test(model)
