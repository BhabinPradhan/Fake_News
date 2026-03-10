import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms, models
import torch.nn.functional as F
import os

class MopedInference:
    def __init__(self, model_path, dataset_type='english', device=None):
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dataset_type = dataset_type
        
        # 1. DYNAMIC ARCHITECTURE SELECTION
        if dataset_type == 'weibo':
            # Weibo has unique layer sizes (e.g., fc1 = 200)
            from MoPeD_weibo import MoPeD, Config
            print(f"Initializing MoPeD Weibo Architecture")
            # We must map the config class to a format the wrapper understands
            self.config = Config()
            # In your weibo script, the class is called 'MoPeD'
            from MoPeD_weibo import MoPeD as ModelClass
            
        elif 'snopes' in model_path.lower() or 'mmhl' in model_path.lower():
            from MoPeD_snopes import MoPeD_MMHL, Config
            print(f"Initializing MoPeD Snopes/MMHL Architecture")
            self.config = Config()
            ModelClass = MoPeD_MMHL
            
        else:
            # Default to XFacta (fc1 = 128)
            from MoPeD_xfacta import MoPeD_MMHL, Config
            print(f"Initializing MoPeD XFacta Architecture")
            self.config = Config()
            ModelClass = MoPeD_MMHL
            
        from transformers import BertTokenizer
        bert_path = 'google-bert/bert-base-chinese' if dataset_type == 'weibo' else 'bert-base-uncased'
        self.tokenizer = BertTokenizer.from_pretrained(bert_path)
        
        # 2. Initialize the model shell
        # 3. Load Weights (must happen BEFORE building Weibo model so we can read shapes)
        print(f"Loading MoPeD weights from: {model_path}")
        state_dict = torch.load(model_path, map_location=self.device)
        sd = state_dict.get('model_state_dict', state_dict)

        # If it's Weibo, read architecture shapes directly from the checkpoint
        # so the model shell always matches what was trained — no hardcoding.
        if dataset_type == 'weibo':
            vocab_size, embed_dim = sd['word_embedding.weight'].shape
            kernel_sizes = sorted([
                w.shape[2] for k, w in sd.items()
                if k.startswith('convs.') and k.endswith('.weight')
            ])
            # Derive modality width and auxiliary feature width from checkpoint.
            # att_feature concatenates four modality vectors before fc5.
            self.weibo_modality_dim = sd['fc5.weight'].shape[1] // 4
            self.weibo_aux_dim = self.weibo_modality_dim - 300 - 50
            if self.weibo_aux_dim <= 0:
                raise ValueError(
                    f"Invalid Weibo aux dim derived from checkpoint: {self.weibo_aux_dim}"
                )
            print(f"  Checkpoint shapes: vocab={vocab_size}, embed_dim={embed_dim}, kernels={kernel_sizes}")
            weibo_dict_config = {
                'maxlen': 170,
                'num_classes': 2,
                'target_names': ['Real', 'Fake'],
                'kernel_sizes': kernel_sizes,
                'user_self_attention': False,
                'embedding_weights': torch.zeros(vocab_size, embed_dim),
                'X_img': [torch.zeros(2048)]
            }
            self.model = ModelClass(weibo_dict_config).to(self.device)
        else:
            self.model = ModelClass(self.config, self.tokenizer).to(self.device)

        # Load weights into the now correctly-shaped model
        if 'model_state_dict' in state_dict:
            self.model.load_state_dict(state_dict['model_state_dict'])
        else:
            self.model.load_state_dict(sd)
            
        self.model.eval()

        # Image Feature Extractor
        self.resnet = models.resnet50(pretrained=True)
        self.resnet.eval()
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-1]).to(self.device)
        
        self.img_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def predict(self, text, image_pil):
        img_t = self.img_transform(image_pil).unsqueeze(0).to(self.device)
        with torch.no_grad():
            image_feature = self.resnet(img_t).squeeze()
            # Always keep batch dimension — squeeze() collapses it at batch_size=1
            if image_feature.dim() == 1:
                image_feature = image_feature.unsqueeze(0)
            # Weibo checkpoints expect an auxiliary image width (commonly 512),
            # not raw ResNet 2048. Adapt width to match trained checkpoint shape.
            if self.dataset_type == 'weibo' and hasattr(self, 'weibo_aux_dim'):
                expected_dim = int(self.weibo_aux_dim)
                if image_feature.size(1) != expected_dim:
                    image_feature = F.adaptive_avg_pool1d(
                        image_feature.unsqueeze(1),
                        expected_dim
                    ).squeeze(1)
            image_feature = image_feature.to(self.device)  # ← pin to correct GPU

        tokens = self.tokenizer(text, padding='max_length', truncation=True,
                                max_length=170, return_tensors='pt')
        ids  = tokens['input_ids'].to(self.device)
        mask = tokens['attention_mask'].to(self.device)

        with torch.no_grad():
            if hasattr(self.model, 'word_embedding'):
                ids = ids.clamp(0, self.model.word_embedding.num_embeddings - 1)
                bsz = ids.size(0)
                dummy_tid = torch.zeros(bsz, dtype=torch.long).to(self.device)
                text_feat_dim = int(getattr(self, 'weibo_aux_dim', 512))
                dummy_text_feat = torch.zeros(bsz, text_feat_dim).to(self.device)
                logits = self.model(dummy_tid, ids, image_feature, dummy_text_feat)
            else:
                logits = self.model(ids, mask, image_feature)

            probs = F.softmax(logits, dim=1).squeeze().cpu().numpy()

        return {"Real": float(probs[0]), "Fake": float(probs[1])}
