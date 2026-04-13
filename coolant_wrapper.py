import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from transformers import BertTokenizer
import torch.nn.functional as F
import os

from mymodel import Multi_Model
from clip import CLIP

# This class will serve as a wrapper around the COOLANT model, allowing us to load it and run inference in a consistent way with our ensemble. 
# It handles the specific architecture and preprocessing requirements of COOLANT, and provides a simple predict() method that takes text and an image and returns Real/Fake probabilities. 
# This will allow us to easily integrate COOLANT into our ModelManager ensemble without needing to modify the core logic there.
class CoolantInference:
    def __init__(self, model_path, device=None):
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.bert_path = "bert-base-uncased"
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_path)
        
        # COOLANT usually uses a CLIP module for alignment and a Multi_Model for detection
        self.clip_mod = CLIP(64, self.bert_path).to(self.device)
        self.det_mod = Multi_Model(self.bert_path).to(self.device)
        
        # Load Weights 
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Matches Weibo and Med-MMHL formats
        if 'det_state_dict' in checkpoint: 
            self.det_mod.load_state_dict(checkpoint['det_state_dict'])
            self.clip_mod.load_state_dict(checkpoint['clip_state_dict'])
            
        # Matches XFacta and Snopes formats
        elif 'det' in checkpoint: 
            self.det_mod.load_state_dict(checkpoint['det'])
            self.clip_mod.load_state_dict(checkpoint['clip'])
        
        else:
            raise KeyError(f"Could not find recognized keys in {model_path}. Found: {list(checkpoint.keys())}")
            
        self.det_mod.eval()
        self.clip_mod.eval()

        # 3. Standard Preprocessing 
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def predict(self, text, image_pil):
        """
        Inputs: Text string, PIL Image
        Output: {'Real': prob, 'Fake': prob}
        """
        # Preprocess Image
        img_tensor = self.transform(image_pil).unsqueeze(0).to(self.device)
        
        # Preprocess Text
        tokens = self.tokenizer(text, padding='max_length', truncation=True, 
                                max_length=170, return_tensors='pt')
        t0 = tokens['input_ids'].to(self.device)
        t1 = tokens['token_type_ids'].to(self.device)
        t2 = tokens['attention_mask'].to(self.device)

        with torch.no_grad():
            # Alignment pass which is required by COOLANT
            i_aln, t_aln = self.clip_mod(t0, t1, t2, img_tensor)
            
            # Detection pass
            output, _, _ = self.det_mod(t0, t1, t2, img_tensor, t_aln, i_aln)
            
            # Convert to probabilities
            probs = F.softmax(output, dim=1).squeeze().cpu().numpy()
            
        return {
            "Real": float(probs[0]),
            "Fake": float(probs[1])
        }

if __name__ == "__main__":
    pass
