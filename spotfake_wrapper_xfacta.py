import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms, models
from transformers import BertTokenizer, BertModel
import torch.nn.functional as F

class SpotFakeModelXFacta(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = BertModel.from_pretrained('bert-base-chinese')
        resnet = models.resnet50(weights='DEFAULT')
        self.resnet = nn.Sequential(*list(resnet.children())[:-1])
        self.fc = nn.Linear(768 + 2048, 2)

    def forward(self, ids, mask, imgs):
        t = self.bert(ids, mask).pooler_output
        v = self.resnet(imgs).squeeze()
        if len(v.shape) == 1: v = v.unsqueeze(0)
        return self.fc(torch.cat([t, v], 1))

class SpotFakeInferenceXFacta:
    def __init__(self, model_path="/home/vongoct/checkpoints/spotfake_xfacta.pth", device=None):
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
        self.model = SpotFakeModelXFacta().to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

    def predict(self, text, image_pil):
        enc = self.tokenizer(text, max_length=256, padding='max_length', truncation=True, return_tensors='pt')
        ids, mask = enc['input_ids'].to(self.device), enc['attention_mask'].to(self.device)
        img_t = self.transform(image_pil).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(ids, mask, img_t)
            probs = F.softmax(logits, dim=1).squeeze().cpu().numpy()
        return {'Real': float(probs[1]), 'Fake': float(probs[0])}
