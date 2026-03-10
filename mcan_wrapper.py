import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision import transforms, models
from transformers import BertTokenizer, BertModel
from scipy.fftpack import fft, dct
from PIL import Image

# ============================================================================
# HELPER: FREQUENCY DOMAIN (DCT) PROCESSING
# ============================================================================
def process_dct_img(img_tensor):
    """Replicates the DCT/FFT processing used in your training scripts."""
    img = img_tensor.numpy()
    height, width = img.shape[1], img.shape[2]
    N = 8
    step = int(height / N)
    dct_img = np.zeros((1, N*N, step*step, 1), dtype=np.float32)
    fft_img = np.zeros((1, N*N, step*step, 1))
    i = 0
    for row in np.arange(0, height, step):
        for col in np.arange(0, width, step):
            block = np.array(img[:, row:(row+step), col:(col+step)], dtype=np.float32)
            block1 = block.reshape(-1, step*step, 1)
            dct_img[:, i, :, :] = dct(block1)
            i += 1
    fft_img[:, :, :, :] = fft(dct_img[:, :, :, :]).real
    fft_img = torch.from_numpy(fft_img).float()
    new_img = F.interpolate(fft_img, size=[250, 1])
    return new_img.squeeze(0).squeeze(-1)


# ============================================================================
# SHARED COMPONENTS (xfacta / snopes / mmhl)
# ============================================================================

class vgg(nn.Module):
    def __init__(self):
        super(vgg, self).__init__()
        vgg_19 = models.vgg19(weights='DEFAULT')
        self.feature = vgg_19.features
        self.classifier = nn.Sequential(*list(vgg_19.classifier.children())[:-3])

    def forward(self, img):
        return self.classifier(self.feature(img).view(img.size(0), -1))


class DctCNN(nn.Module):
    def __init__(self):
        super(DctCNN, self).__init__()
        self.simple_conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten()
        )
        self.project = nn.Linear(63488, 4096)

    def forward(self, x):
        return self.project(self.simple_conv(x.unsqueeze(1)))


# ── Standard multi-head attention (xfacta / snopes / mmhl) ──────────────────
class MultiHeadAttention(nn.Module):
    def __init__(self, model_dim=256, num_heads=8, dropout=0.5):
        super(MultiHeadAttention, self).__init__()
        self.model_dim    = model_dim
        self.num_heads    = num_heads
        self.dim_per_head = model_dim // num_heads
        self.linear_k     = nn.Linear(model_dim, model_dim, bias=False)
        self.linear_v     = nn.Linear(model_dim, model_dim, bias=False)
        self.linear_q     = nn.Linear(model_dim, model_dim, bias=False)
        self.linear_final = nn.Linear(model_dim, model_dim, bias=False)
        self.dropout      = nn.Dropout(dropout)
        self.layer_norm   = nn.LayerNorm(model_dim)

    def forward(self, query, key, value):
        batch_size = query.size(0)
        q = self.linear_q(query).view(batch_size, self.num_heads, self.dim_per_head)
        k = self.linear_k(key).view(batch_size, self.num_heads, self.dim_per_head)
        v = self.linear_v(value).view(batch_size, self.num_heads, self.dim_per_head)
        dist   = torch.matmul(q, k.transpose(-2, -1)) / (self.dim_per_head ** 0.5)
        output = torch.matmul(F.softmax(dist, dim=-1), v).view(batch_size, self.model_dim)
        return self.layer_norm(query + self.dropout(self.linear_final(output)))


class multimodal_fusion_layer(nn.Module):
    """Standard fusion layer — matches xfacta / snopes / mmhl checkpoints."""
    def __init__(self, model_dim=256, num_heads=8, dropout=0.5):
        super(multimodal_fusion_layer, self).__init__()
        self.at1 = MultiHeadAttention(model_dim, num_heads, dropout)
        self.at2 = MultiHeadAttention(model_dim, num_heads, dropout)
        self.fusion_linear = nn.Linear(model_dim * 2, model_dim)

    def forward(self, x1, x2):
        return self.fusion_linear(torch.cat([self.at1(x1, x2, x2), self.at2(x2, x1, x1)], dim=1))


# ============================================================================
# WEIBO-SPECIFIC COMPONENTS
# Transcribed exactly from mcan_weibo.py (NetShareFusion training script).
#
# Key structural differences vs english variants:
#   - MultiHeadAttention projects from dim 1 (scalar per position) not model_dim
#     linear_k/v/q : nn.Linear(1, model_dim)  → weight (256, 1)
#     linear_final  : nn.Linear(model_dim, 1) → weight (1, 256)
#   - Each fusion layer adds PositionalWiseFeedForward blocks (feed_forward_1/2)
#   - Classifier is linear1/linear2 at top level, not a Sequential named classifier
# ============================================================================

class WeiboMultiHeadAttention(nn.Module):
    """
    Exact copy of MultiHeadAttention from mcan_weibo.py.
    Projects each of the model_dim positions from scalar (size 1) to full dim,
    splits into num_heads, runs scaled dot-product attention, projects back to 1.
    """
    def __init__(self, model_dim=256, num_heads=4, dropout=0.5):
        super(WeiboMultiHeadAttention, self).__init__()
        self.model_dim    = model_dim
        self.num_heads    = num_heads
        self.dim_per_head = model_dim // num_heads          # 256 // 4 = 64
        # Linear(1, 256) → stored weight shape (256, 1) — matches checkpoint
        self.linear_k     = nn.Linear(1, self.dim_per_head * num_heads, bias=False)
        self.linear_v     = nn.Linear(1, self.dim_per_head * num_heads, bias=False)
        self.linear_q     = nn.Linear(1, self.dim_per_head * num_heads, bias=False)
        # Linear(256, 1) → stored weight shape (1, 256) — matches checkpoint
        self.linear_final = nn.Linear(model_dim, 1, bias=False)
        self.dropout      = nn.Dropout(dropout)
        self.layer_norm   = nn.LayerNorm(model_dim)

    def forward(self, query, key, value):
        # query/key/value: (B, 256)
        residual = query
        # Unsqueeze to (B, 256, 1) so Linear(1→256) projects each scalar position
        k = self.linear_k(key.unsqueeze(-1))        # (B, 256, 256)
        v = self.linear_v(value.unsqueeze(-1))      # (B, 256, 256)
        q = self.linear_q(query.unsqueeze(-1))      # (B, 256, 256)
        # Split into heads: (B, num_heads, model_dim, dim_per_head)
        k = k.view(-1, self.num_heads, self.model_dim, self.dim_per_head)
        v = v.view(-1, self.num_heads, self.model_dim, self.dim_per_head)
        q = q.view(-1, self.num_heads, self.model_dim, self.dim_per_head)
        # Scale matches training: (dim_per_head // num_heads) ** -0.5 = 16 ** -0.5
        scale      = (self.dim_per_head // self.num_heads) ** -0.5
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * scale   # (B, H, 256, 256)
        attn        = F.softmax(attn_scores, dim=-1)
        attn        = self.dropout(attn)
        out         = torch.matmul(attn, v)                           # (B, H, 256, 64)
        # Merge heads back: (B, 256, dim_per_head * num_heads) = (B, 256, 256)
        out = out.view(-1, self.model_dim, self.dim_per_head * self.num_heads)
        out = self.linear_final(out).squeeze(-1)                      # (B, 256)
        out = self.dropout(out)
        return self.layer_norm(residual + out)


class WeiboFFN(nn.Module):
    """
    Exact copy of PositionalWiseFeedForward from mcan_weibo.py.
    Key names w1/w2/layer_norm match the feed_forward_1/2 entries in the checkpoint.
    Dropout is a no-op at eval() time so omitting it from the state_dict is fine.
    """
    def __init__(self, model_dim=256, ff_dim=2048, dropout=0.5):
        super(WeiboFFN, self).__init__()
        self.w1         = nn.Linear(model_dim, ff_dim)
        self.w2         = nn.Linear(ff_dim, model_dim)
        self.dropout    = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(model_dim)

    def forward(self, x):
        residual = x
        x = self.dropout(self.w2(F.relu(self.w1(x))))
        return self.layer_norm(residual + x)


class multimodal_fusion_layer_weibo(nn.Module):
    """
    Weibo fusion layer — matches checkpoint key structure:
      attention_1, attention_2, feed_forward_1, feed_forward_2, fusion_linear
    """
    def __init__(self, model_dim=256, num_heads=4, ff_dim=2048, dropout=0.5):
        super(multimodal_fusion_layer_weibo, self).__init__()
        self.attention_1    = WeiboMultiHeadAttention(model_dim, num_heads, dropout)
        self.attention_2    = WeiboMultiHeadAttention(model_dim, num_heads, dropout)
        self.feed_forward_1 = WeiboFFN(model_dim, ff_dim, dropout)
        self.feed_forward_2 = WeiboFFN(model_dim, ff_dim, dropout)
        self.fusion_linear  = nn.Linear(model_dim * 2, model_dim)

    def forward(self, x1, x2):
        a1 = self.feed_forward_1(self.attention_1(x1, x2, x2))
        a2 = self.feed_forward_2(self.attention_2(x2, x1, x1))
        return self.fusion_linear(torch.cat([a1, a2], dim=1))


# ============================================================================
# MAIN INFERENCE CLASS
# ============================================================================

class McanInference(nn.Module):
    def __init__(self, model_path, dataset_type='english', device=None):
        super(McanInference, self).__init__()
        self.device       = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dataset_type = dataset_type

        # ── BERT variant ────────────────────────────────────────────────────
        self.bert_name = 'google-bert/bert-base-chinese' if dataset_type == 'weibo' else 'bert-base-uncased'
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_name)

        # ── Shared backbone ──────────────────────────────────────────────────
        model_dim        = 256
        self.bert        = BertModel.from_pretrained(self.bert_name)
        self.linear_text = nn.Linear(768, model_dim)
        self.vgg         = vgg()
        self.dct_img     = DctCNN()
        self.linear_image = nn.Linear(4096, model_dim)
        self.linear_dct   = nn.Linear(4096, model_dim)

        # ── Architecture fork: weibo vs english ─────────────────────────────
        # Weibo checkpoint keys:
        #   fusion_layers.N.attention_1 / attention_2 / feed_forward_1 / feed_forward_2
        #   linear1, linear2   (top-level, not self.classifier)
        #
        # English checkpoint keys:
        #   fusion_layers.N.at1 / at2
        #   classifier.0, classifier.3  (Sequential)

        if dataset_type == 'weibo':
            self.fusion_layers = nn.ModuleList(
                [multimodal_fusion_layer_weibo(model_dim, num_heads=4, ff_dim=2048, dropout=0.5) for _ in range(2)]
            )
            # Named linear1 / linear2 to match weibo checkpoint top-level keys
            self.linear1 = nn.Linear(model_dim, 35)
            self.linear2 = nn.Linear(35, 2)
        else:
            self.fusion_layers = nn.ModuleList(
                [multimodal_fusion_layer(model_dim, num_heads=4, dropout=0.5) for _ in range(2)]
            )
            # Named classifier.0 / classifier.3 to match english checkpoint keys
            self.classifier = nn.Sequential(
                nn.Linear(model_dim, 35), nn.ReLU(), nn.Dropout(0.5), nn.Linear(35, 2)
            )

        # ── Load weights ─────────────────────────────────────────────────────
        print(f"Loading MCAN weights from: {model_path}")
        self.load_state_dict(torch.load(model_path, map_location=self.device))
        self.to(self.device)
        self.eval()

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

    def _classify(self, feat):
        """Route through the correct classifier head depending on variant."""
        if self.dataset_type == 'weibo':
            return self.linear2(F.relu(self.linear1(feat)))
        else:
            return self.classifier(feat)

    def predict(self, text, image_pil):
        # ── Image features ───────────────────────────────────────────────────
        vgg_in  = self.transform(image_pil.convert('RGB')).unsqueeze(0).to(self.device)
        dct_in  = self.transform(image_pil.convert('L'))
        dct_feat = process_dct_img(dct_in).unsqueeze(0).to(self.device)

        # ── Text tokens ──────────────────────────────────────────────────────
        tokens = self.tokenizer(
            text, padding='max_length', truncation=True, max_length=160, return_tensors='pt'
        )
        t_ids  = tokens['input_ids'].to(self.device)
        t_mask = tokens['attention_mask'].to(self.device)
        t_type = tokens['token_type_ids'].to(self.device)

        with torch.no_grad():
            t_out = F.relu(self.linear_text(
                self.bert(t_ids, attention_mask=t_mask, token_type_ids=t_type)[1]
            ))
            v_out = F.relu(self.linear_image(self.vgg(vgg_in)))
            d_out = F.relu(self.linear_dct(self.dct_img(dct_feat)))

            feat = v_out
            for layer in self.fusion_layers:
                feat = layer(feat, d_out)
            for layer in self.fusion_layers:
                feat = layer(feat, t_out)

            probs = F.softmax(self._classify(feat), dim=1).squeeze().cpu().numpy()

        return {"Real": float(probs[0]), "Fake": float(probs[1])}