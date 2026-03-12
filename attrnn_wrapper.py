import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel
from torchvision import models, transforms
from PIL import Image

class AttRNNModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        # Matches the GPU 2 training logic
        vgg = models.vgg19(weights=None)
        self.vgg = vgg.features
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.clf = nn.Linear(768 + 512, 2)

    def forward(self, ids, msk, img):
        t_feat = self.bert(ids, msk).pooler_output
        i_feat = torch.flatten(self.gap(self.vgg(img)), 1)
        return self.clf(torch.cat((t_feat, i_feat), dim=1))

class AttRNNInference:
    def __init__(self, model_path, device="cpu"):
        self.device = device
        self.model = AttRNNModel().to(self.device)
        # Load the weights you just generated
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
        
        return {"Real": float(probs[0]), "Fake": float(probs[1])}
