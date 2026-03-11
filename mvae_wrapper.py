import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel
from torchvision import models, transforms
from PIL import Image

class MVAEModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        vgg = models.vgg19(weights=None)
        self.vgg = vgg.features
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        
        # Matches the GPU 3 training logic exactly
        self.text_enc = nn.Linear(768, 256)
        self.img_enc = nn.Linear(512, 256)
        self.clf = nn.Linear(256, 2)

    def forward(self, ids, msk, img):
        t_feat = self.text_enc(self.bert(ids, msk).pooler_output)
        img_out = self.vgg(img)
        i_feat = self.img_enc(torch.flatten(self.gap(img_out), 1))
        # Joint product fusion as used in training
        joint = t_feat * i_feat 
        return self.clf(joint)

class MVAEInference:
    def __init__(self, model_path, device="cpu"):
        self.device = device
        self.model = MVAEModel().to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.tfm = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def predict(self, text, image_pil):
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=256).to(self.device)
        img_t = self.tfm(image_pil).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            logits = self.model(inputs['input_ids'], inputs['attention_mask'], img_t)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        
        return {"Real": float(probs[1]), "Fake": float(probs[0])}
