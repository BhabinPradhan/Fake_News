import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel
from torchvision import models, transforms

class AttRNNModelWeibo(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = BertModel.from_pretrained('bert-base-chinese')
        resnet = models.resnet50(weights='DEFAULT')
        self.resnet = nn.Sequential(*list(resnet.children())[:-1])
        self.w_v = nn.Linear(2048, 768)
        self.fc = nn.Linear(768, 2)

    def forward(self, ids, mask, imgs):
        t = self.bert(ids, mask).pooler_output
        v = self.resnet(imgs).squeeze()
        if len(v.shape) == 1: v = v.unsqueeze(0)
        v_transformed = torch.tanh(self.w_v(v))
        combined = t * v_transformed
        return self.fc(combined)

class AttRNNInferenceWeibo:
    def __init__(self, model_path="/home/vongoct/checkpoints/attrnn_weibo.pth", device="cpu"):
        self.device = device
        self.model = AttRNNModelWeibo().to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
        self.tfm = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])

    def predict(self, text, image_pil):
        enc = self.tokenizer(text, return_tensors="pt", padding='max_length', truncation=True, max_length=256)
        ids, mask = enc['input_ids'].to(self.device), enc['attention_mask'].to(self.device)
        img_t = self.tfm(image_pil).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(ids, mask, img_t)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        return {"Real": float(probs[0]), "Fake": float(probs[1])}
