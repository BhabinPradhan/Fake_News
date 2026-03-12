import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms, models
from transformers import BertTokenizer, BertModel
import torch.nn.functional as F

class SpotFakeModel(nn.Module):
    def __init__(self):
        super(SpotFakeModel, self).__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        vgg = models.vgg19(weights="VGG19_Weights.IMAGENET1K_V1")
        self.vgg_features = vgg.features
        self.vgg_avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.text_fc = nn.Linear(768, 128)
        self.img_fc = nn.Linear(512, 128)
        self.classifier = nn.Sequential(nn.Linear(256, 64), nn.ReLU(), nn.Dropout(0.2), nn.Linear(64, 2))

    def forward(self, input_ids, attention_mask, images):
        text_out = self.bert(input_ids=input_ids, attention_mask=attention_mask).pooler_output
        text_feat = self.text_fc(text_out)
        img_out = self.vgg_features(images)
        img_out = self.vgg_avgpool(img_out)
        img_feat = self.img_fc(torch.flatten(img_out, 1))
        return self.classifier(torch.cat((text_feat, img_feat), dim=1))

class SpotFakeInference:
    def __init__(self, model_path, device=None):
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.model = SpotFakeModel().to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def predict(self, text, image_pil):
        enc = self.tokenizer(text, max_length=256, padding='max_length', truncation=True, return_tensors='pt')
        ids, mask = enc['input_ids'].to(self.device), enc['attention_mask'].to(self.device)
        img_t = self.transform(image_pil).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(ids, mask, img_t)
            probs = F.softmax(logits, dim=1).squeeze().cpu().numpy()
        return {'Fake': float(probs[1]), 'Real': float(probs[0])}
