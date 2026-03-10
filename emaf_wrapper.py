import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from transformers import BertTokenizer
import torch.nn.functional as F
import os
import spacy

# Import your actual model architecture
from emaf_model import EMAF_Model

# This class will serve as a wrapper around the EMAF model, allowing us to load it and run inference in a consistent way with our ensemble. 
# It handles the specific architecture and preprocessing requirements of EMAF, and provides a simple predict() method that takes text and an image and returns Real/Fake probabilities. 
class EmafInference:
    def __init__(self, model_path, lang='en', device=None):
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.lang = lang
        
        #Setup Language Extractor in case we need to extract entities (EMAF does this as part of its architecture)
        if self.lang == 'zh':
            import jieba.posseg as pseg
            self.extractor = pseg
            self.bert_path = "google-bert/bert-base-chinese"
        else:
            try:
                self.extractor = spacy.load("en_core_web_sm")
            except:
                os.system("python -m spacy download en_core_web_sm")
                self.extractor = spacy.load("en_core_web_sm")
            self.bert_path = "bert-base-uncased"
            
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_path)
        
        # The, initialize the model architecture 
        self.model = EMAF_Model(self.bert_path).to(self.device)
        
        # Load Weights 
        checkpoint = torch.load(model_path, map_location=self.device)
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
            
        self.model.eval()

        #Preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def _extract_entities(self, text, max_entities=100):
        """Extracts entities based on the language of the model."""
        if self.lang == 'zh':
            words = self.extractor.cut(text)
            entities = [w.word for w in words if w.flag in ['nr', 'ns', 'nt', 'nz']]
            if not entities:
                import jieba
                entities = list(jieba.cut(text))
        else:
            doc = self.extractor(text)
            entities = [token.text for token in doc if token.pos_ in ['NOUN', 'PROPN']]
            if not entities:
                entities = [token.text for token in doc if not token.is_stop and not token.is_punct]
        
        entities = entities[:max_entities]
        while len(entities) < max_entities:
            entities.append('[PAD]')
        return entities

    def predict(self, text, image_pil):
        # Extract and Tokenize Entities
        entities = self._extract_entities(text)
        entity_ids, entity_masks = [], []
        
        for e in entities:
            tok = self.tokenizer(e, padding='max_length', truncation=True, 
                                max_length=10, return_tensors='pt')
            entity_ids.append(tok['input_ids'])
            entity_masks.append(tok['attention_mask'])

        b_ids = torch.cat(entity_ids, dim=0).unsqueeze(0).to(self.device) # [1, 100, 10]
        b_mask = torch.cat(entity_masks, dim=0).unsqueeze(0).to(self.device)
        b_img = self.transform(image_pil).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(b_ids, b_mask, b_img)
            probs = F.softmax(outputs, dim=1).squeeze().cpu().numpy()
            
        return {"Real": float(probs[0]), "Fake": float(probs[1])}