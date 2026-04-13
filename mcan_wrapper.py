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
# SHARED COMPONENTS
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
    """
    V1 fusion layer — used by all checkpoints (xfacta, snopes, mmhl, weibo retrain).
    Checkpoint keys: fusion_layers.N.at1/at2/fusion_linear
    """
    def __init__(self, model_dim=256, num_heads=8, dropout=0.5):
        super(multimodal_fusion_layer, self).__init__()
        self.at1 = MultiHeadAttention(model_dim, num_heads, dropout)
        self.at2 = MultiHeadAttention(model_dim, num_heads, dropout)
        self.fusion_linear = nn.Linear(model_dim * 2, model_dim)

    def forward(self, x1, x2):
        return self.fusion_linear(torch.cat([self.at1(x1, x2, x2), self.at2(x2, x1, x1)], dim=1))


# ============================================================================
# MAIN INFERENCE CLASS
#
# All four checkpoints (xfacta, snopes, mmhl, weibo) now use identical V1
# architecture. dataset_type only controls which BERT tokenizer is loaded:
#   - 'weibo'   → bert-base-chinese  (Chinese text)
#   - 'english' → bert-base-uncased  (English text)
# ============================================================================

class McanInference(nn.Module):
    def __init__(self, model_path, dataset_type='english', device=None):
        super(McanInference, self).__init__()
        self.device       = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dataset_type = dataset_type

        # Select BERT variant based on language
        self.bert_name = 'google-bert/bert-base-chinese' if dataset_type == 'weibo' else 'bert-base-uncased'
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_name)

        # Shared backbone — identical across all datasets
        model_dim         = 256
        self.bert         = BertModel.from_pretrained(self.bert_name)
        self.linear_text  = nn.Linear(768, model_dim)
        self.vgg          = vgg()
        self.dct_img      = DctCNN()
        self.linear_image = nn.Linear(4096, model_dim)
        self.linear_dct   = nn.Linear(4096, model_dim)

        # V1 fusion layers — same architecture for all checkpoints
        self.fusion_layers = nn.ModuleList(
            [multimodal_fusion_layer(model_dim, num_heads=4, dropout=0.5) for _ in range(2)]
        )

        # Classifier — nn.Sequential matches checkpoint keys classifier.0 / classifier.3
        self.classifier = nn.Sequential(
            nn.Linear(model_dim, 35), nn.ReLU(), nn.Dropout(0.5), nn.Linear(35, 2)
        )

        # Load weights and move to device
        self.load_state_dict(torch.load(model_path, map_location=self.device))
        self.to(self.device)
        self.eval()

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

    def predict(self, text, image_pil):
        # Image features
        vgg_in   = self.transform(image_pil.convert('RGB')).unsqueeze(0).to(self.device)
        dct_in   = self.transform(image_pil.convert('L'))
        dct_feat = process_dct_img(dct_in).unsqueeze(0).to(self.device)

        # Text tokens
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

            probs = F.softmax(self.classifier(feat), dim=1).squeeze().cpu().numpy()

        return {"Real": float(probs[0]), "Fake": float(probs[1])}
