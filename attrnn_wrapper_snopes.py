import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer
from torchvision import models, transforms

class AttRNNInferenceSnopes:
    def __init__(self, model_path, device="cpu"):
        self.device = device
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')  
        self.bert = BertModel.from_pretrained('bert-base-uncased') 
        self.resnet = models.resnet50(pretrained=True)
        self.resnet.fc = nn.Identity()
        self.attention = nn.MultiheadAttention(embed_dim=768, num_heads=8, batch_first=True)
        self.img_proj = nn.Linear(2048, 768)
        self.classifier = nn.Linear(768, 2)
        state = torch.load(model_path, map_location=device)
        self.classifier.load_state_dict(state, strict=False)
        # Ensure all submodules live on the same device
        self.bert.to(device).eval()
        self.resnet.to(device).eval()
        self.attention.to(device).eval()
        self.img_proj.to(device).eval()
        self.classifier.to(device).eval()

    def predict(self, text, image):
        t_input = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128).to(self.device)
        transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
        i_input = transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            t_feat = self.bert(**t_input).last_hidden_state
            i_feat = self.resnet(i_input).unsqueeze(1)
            i_feat_proj = self.img_proj(i_feat)
            attn_output, _ = self.attention(i_feat_proj, t_feat, t_feat)
            output = self.classifier(attn_output.squeeze(1))
            probs = torch.softmax(output, dim=1)[0]
        return {"Real": float(probs[0]), "Fake": float(probs[1])}
