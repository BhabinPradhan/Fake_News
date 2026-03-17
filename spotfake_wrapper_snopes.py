import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer
from torchvision import models, transforms

class SpotFakeInferenceSnopes:
    def __init__(self, model_path, device="cpu"):
        self.device = device
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased') 
        self.bert = BertModel.from_pretrained('bert-base-uncased') 
        self.resnet = models.resnet50(pretrained=True)
        self.resnet.fc = nn.Identity()
        self.classifier = nn.Sequential(
            nn.Linear(768 + 2048, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 2)
        ).to(device)
        state = torch.load(model_path, map_location=device)
        self.classifier.load_state_dict(state, strict=False)
        self.classifier.to(device).eval()
        self.bert.to(device).eval()
        self.resnet.to(device).eval()

    def predict(self, text, image):
        t_input = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128).to(self.device)
        transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
        i_input = transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            t_feat = self.bert(**t_input).pooler_output
            i_feat = self.resnet(i_input)
            combined = torch.cat((t_feat, i_feat), dim=1)
            output = self.classifier(combined)
            probs = torch.softmax(output, dim=1)[0]
        return {"Real": float(probs[0]), "Fake": float(probs[1])}
