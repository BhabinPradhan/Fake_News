import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel
from torchvision import models, transforms

class MVAEModelXFacta(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = BertModel.from_pretrained('bert-base-chinese')
        resnet = models.resnet50(weights='DEFAULT')
        self.resnet = nn.Sequential(*list(resnet.children())[:-1])
        self.fc_mu = nn.Linear(768 + 2048, 512)
        self.fc_logvar = nn.Linear(768 + 2048, 512)
        self.classifier = nn.Linear(512, 2)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, ids, mask, imgs):
        t = self.bert(ids, mask).pooler_output
        v = self.resnet(imgs).squeeze()
        if len(v.shape) == 1: v = v.unsqueeze(0)
        combined = torch.cat([t, v], 1)
        mu, logvar = self.fc_mu(combined), self.fc_logvar(combined)
        z = self.reparameterize(mu, logvar)
        return self.classifier(z)

class MVAEInferenceXFacta:
    def __init__(self, model_path="/home/vongoct/checkpoints/mvae_xfacta.pth", device="cpu"):
        self.device = device
        self.model = MVAEModelXFacta().to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
        self.tfm = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

    def predict(self, text, image_pil):
        enc = self.tokenizer(text, return_tensors="pt", padding='max_length', truncation=True, max_length=256)
        ids = enc['input_ids'].to(self.device)
        mask = enc['attention_mask'].to(self.device)
        img_t = self.tfm(image_pil).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(ids, mask, img_t)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

        return {"Real": float(probs[1]), "Fake": float(probs[0])}
