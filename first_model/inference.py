"""
Multi-Modal Fusion Model Inference
Handles text input from terminal and image from file path
"""

import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from PIL import Image
import torchvision.transforms as transforms
from transformers import AutoTokenizer, AutoModel, AutoConfig

# Import model architecture from training
import sys
sys.path.append(str(Path(__file__).parent))

# ============================================================================
# MODEL ARCHITECTURE (same as training)
# ============================================================================

class MultiModalFusionModel(nn.Module):
    """Fusion model combining text, image, and object detection features"""
    
    def __init__(self, text_dim=768, image_dim=1280, obj_dim=7, 
                 hidden_dim=512, dropout=0.3):
        super(MultiModalFusionModel, self).__init__()
        
        self.text_dim = text_dim
        self.image_dim = image_dim
        self.obj_dim = obj_dim
        
        # Modality-specific processing
        self.text_projection = nn.Sequential(
            nn.Linear(text_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.image_projection = nn.Sequential(
            nn.Linear(image_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.obj_projection = nn.Sequential(
            nn.Linear(obj_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Fusion layers
        fusion_input_dim = 256 + 256 + 64 + 2  # +2 for modality flags
        
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        # Classification head
        self.classifier = nn.Linear(128, 2)
    
    def forward(self, text_emb, image_emb, obj_emb, has_text, has_image):
        batch_size = text_emb.size(0)
        
        # Project each modality
        text_features = self.text_projection(text_emb)
        image_features = self.image_projection(image_emb)
        obj_features = self.obj_projection(obj_emb)
        
        # Concatenate all features + modality flags
        modality_flags = torch.stack([has_text, has_image], dim=1)
        fused_features = torch.cat([
            text_features, 
            image_features, 
            obj_features,
            modality_flags
        ], dim=1)
        
        # Fusion
        fused = self.fusion(fused_features)
        
        # Classification
        logits = self.classifier(fused)
        
        return logits

# ============================================================================
# EMBEDDING EXTRACTORS
# ============================================================================

class TextEmbeddingExtractor:
    """Extract text embeddings using fine-tuned BERT model"""
    
    def __init__(self, checkpoint_path, device='cuda'):
        self.device = device
        checkpoint_path = Path(checkpoint_path)
        
        # Load config and tokenizer
        self.config = AutoConfig.from_pretrained(str(checkpoint_path))
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        
        # Load model
        from transformers import AutoModelForSequenceClassification
        classifier_model = AutoModelForSequenceClassification.from_pretrained(
            str(checkpoint_path),
            config=self.config,
            use_safetensors=True
        )
        
        # Extract base BERT model for embeddings
        self.model = classifier_model.bert.to(device)
        self.model.eval()
        
        print(f"✓ Text model loaded from {checkpoint_path}")
    
    def extract(self, text, max_length=512):
        """Extract 768-dim embedding from text"""
        encoding = self.tokenizer(
            str(text),
            add_special_tokens=True,
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        
        # Clamp to vocab size
        model_vocab_size = self.model.config.vocab_size
        input_ids = torch.clamp(input_ids, max=model_vocab_size - 1)
        
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True
            )
            # Get [CLS] token embedding
            embedding = outputs.last_hidden_state[:, 0, :]  # (1, 768)
        
        return embedding

class ImageEmbeddingExtractor:
    """Extract image embeddings using EfficientNet + Weight Predictor"""
    
    def __init__(self, weight_predictor_path, device='cuda'):
        self.device = device
        
        # Load EfficientNet-B0
        from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
        self.efficientnet = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        self.efficientnet.classifier = nn.Identity()  # Remove classifier, keep features
        self.efficientnet = self.efficientnet.to(device)
        self.efficientnet.eval()
        
        # Load weight predictor
        self.weight_predictor = self._load_weight_predictor(weight_predictor_path)
        
        # Image transforms
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        print(f"✓ Image model loaded with weight predictor")
    
    def _load_weight_predictor(self, model_path):
        """Load the weight predictor model"""
        class WeightPredictor(nn.Module):
            def __init__(self, embedding_dim=1280, hidden_dim=512):
                super(WeightPredictor, self).__init__()
                
                # Input: embedding (1280) + initial_weight (1) = 1281
                self.fc1 = nn.Linear(embedding_dim + 1, hidden_dim)
                self.bn1 = nn.BatchNorm1d(hidden_dim)
                self.dropout1 = nn.Dropout(0.3)
                
                self.fc2 = nn.Linear(hidden_dim, 256)
                self.bn2 = nn.BatchNorm1d(256)
                self.dropout2 = nn.Dropout(0.3)
                
                self.fc3 = nn.Linear(256, 128)
                self.bn3 = nn.BatchNorm1d(128)
                self.dropout3 = nn.Dropout(0.2)
                
                self.fc_out = nn.Linear(128, 1)
            
            def forward(self, embedding, initial_weight):
                if embedding.dim() == 1:
                    embedding = embedding.unsqueeze(0)
                if initial_weight.dim() == 0:
                    initial_weight = initial_weight.unsqueeze(0).unsqueeze(0)
                elif initial_weight.dim() == 1:
                    initial_weight = initial_weight.unsqueeze(1)
                
                # Concatenate embedding with initial weight
                x = torch.cat([embedding, initial_weight], dim=1)
                
                x = self.fc1(x)
                x = self.bn1(x)
                x = torch.relu(x)
                x = self.dropout1(x)
                
                x = self.fc2(x)
                x = self.bn2(x)
                x = torch.relu(x)
                x = self.dropout2(x)
                
                x = self.fc3(x)
                x = self.bn3(x)
                x = torch.relu(x)
                x = self.dropout3(x)
                
                x = self.fc_out(x)
                return torch.sigmoid(x).squeeze(-1)
        
        model = WeightPredictor().to(self.device)
        # Load state dict directly (not wrapped in a checkpoint dict)
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.eval()
        
        return model
    
    def extract(self, image_path):
        """Extract 1280-dim embedding from image"""
        # Load and transform image
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Get initial weight based on emotion (if we had emotion classifier)
        # For now, use neutral initial weight
        initial_weight = torch.tensor([0.5], dtype=torch.float32).to(self.device)
        
        with torch.no_grad():
            # Get EfficientNet features
            features = self.efficientnet(image_tensor)  # (1, 1280)
            
            # Get adaptive weight
            weight = self.weight_predictor(features.squeeze(0), initial_weight)
            
        return features, weight.item()

class HarmfulObjectDetector:
    """Detects harmful objects from image (requires label file or object detection model)"""
    
    HARMFUL_OBJECTS = {
        'knife': 0.9, 'gun': 1.0, 'rifle': 1.0, 'pistol': 1.0, 'weapon': 1.0,
        'blade': 0.85, 'razor': 0.85, 'pills': 0.7, 'medication': 0.6,
        'drugs': 0.8, 'syringe': 0.75, 'needle': 0.7, 'rope': 0.8,
        'noose': 1.0, 'alcohol': 0.5, 'cigarette': 0.3, 'poison': 0.95,
        'gasoline': 0.7,
    }
    
    def __init__(self, labels_dir):
        self.labels_dir = Path(labels_dir)
    
    def detect(self, image_path):
        """
        Detect harmful objects from label file
        Returns 7-dim feature vector
        """
        # Get corresponding label file
        image_name = Path(image_path).stem
        label_file = self.labels_dir / f"{image_name}.txt"
        
        features = {
            'has_harmful_object': 0,
            'harmful_object_count': 0,
            'max_risk_score': 0.0,
            'avg_risk_score': 0.0,
            'weapon_detected': 0,
            'drug_detected': 0,
            'alcohol_detected': 0,
        }
        
        if not label_file.exists():
            # No label file found - return zeros
            return np.array([
                features['has_harmful_object'],
                features['harmful_object_count'],
                features['max_risk_score'],
                features['avg_risk_score'],
                features['weapon_detected'],
                features['drug_detected'],
                features['alcohol_detected'],
            ], dtype=np.float32)
        
        try:
            with open(label_file, 'r') as f:
                objects = [line.strip().lower() for line in f if line.strip()]
            
            if not objects:
                return np.array([0, 0, 0, 0, 0, 0, 0], dtype=np.float32)
            
            risk_scores = []
            for obj in objects:
                for harmful_obj, risk in self.HARMFUL_OBJECTS.items():
                    if harmful_obj in obj:
                        features['has_harmful_object'] = 1
                        features['harmful_object_count'] += 1
                        risk_scores.append(risk)
                        
                        if harmful_obj in ['gun', 'rifle', 'pistol', 'weapon', 'knife', 'blade']:
                            features['weapon_detected'] = 1
                        elif harmful_obj in ['drugs', 'pills', 'syringe', 'needle']:
                            features['drug_detected'] = 1
                        elif harmful_obj in ['alcohol', 'cigarette']:
                            features['alcohol_detected'] = 1
                        
                        break
            
            if risk_scores:
                features['max_risk_score'] = max(risk_scores)
                features['avg_risk_score'] = np.mean(risk_scores)
        
        except Exception as e:
            print(f"Warning: Error reading {label_file}: {e}")
        
        return np.array([
            features['has_harmful_object'],
            features['harmful_object_count'],
            features['max_risk_score'],
            features['avg_risk_score'],
            features['weapon_detected'],
            features['drug_detected'],
            features['alcohol_detected'],
        ], dtype=np.float32)

# ============================================================================
# INFERENCE PIPELINE
# ============================================================================

class SuicideDetectionInference:
    """Complete inference pipeline for suicide detection"""
    
    def __init__(self, fusion_model_path, text_model_path, image_model_path, 
                 objdec_labels_dir, device='cuda'):
        self.device = device
        
        print("Loading models...")
        print("-" * 80)
        
        # Load fusion model
        self.fusion_model = MultiModalFusionModel().to(device)
        checkpoint = torch.load(fusion_model_path, map_location=device)
        self.fusion_model.load_state_dict(checkpoint['model_state_dict'])
        self.fusion_model.eval()
        print(f"✓ Fusion model loaded (F1: {checkpoint['best_f1']:.4f})")
        
        # Load text extractor
        self.text_extractor = TextEmbeddingExtractor(text_model_path, device)
        
        # Load image extractor
        self.image_extractor = ImageEmbeddingExtractor(image_model_path, device)
        
        # Load object detector
        self.obj_detector = HarmfulObjectDetector(objdec_labels_dir)
        print(f"✓ Object detector ready")
        
        print("-" * 80)
        print("✓ All models loaded successfully!")
        print()
    
    def predict(self, text=None, image_path=None):
        """
        Predict suicide risk from text and/or image
        
        Args:
            text: Text input (optional)
            image_path: Path to image (optional)
        
        Returns:
            dict with prediction results
        """
        # Extract text embedding
        if text:
            text_emb = self.text_extractor.extract(text)
            has_text = torch.tensor([1.0], dtype=torch.float32).to(self.device)
        else:
            text_emb = torch.zeros(1, 768, dtype=torch.float32).to(self.device)
            has_text = torch.tensor([0.0], dtype=torch.float32).to(self.device)
        
        # Extract image embedding and object features
        if image_path:
            image_emb, adaptive_weight = self.image_extractor.extract(image_path)
            obj_features = self.obj_detector.detect(image_path)
            obj_emb = torch.tensor(obj_features, dtype=torch.float32).unsqueeze(0).to(self.device)
            has_image = torch.tensor([1.0], dtype=torch.float32).to(self.device)
        else:
            image_emb = torch.zeros(1, 1280, dtype=torch.float32).to(self.device)
            obj_emb = torch.zeros(1, 7, dtype=torch.float32).to(self.device)
            has_image = torch.tensor([0.0], dtype=torch.float32).to(self.device)
            adaptive_weight = 0.0
            obj_features = np.zeros(7)
        
        # Run fusion model
        with torch.no_grad():
            logits = self.fusion_model(text_emb, image_emb, obj_emb, has_text, has_image)
            probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
        
        # Prepare results
        result = {
            'prediction': 'SUICIDE RISK' if probs[1] > probs[0] else 'NON-SUICIDE',
            'suicide_probability': float(probs[1]),
            'non_suicide_probability': float(probs[0]),
            'confidence': float(max(probs)),
            'modalities_used': {
                'text': bool(text),
                'image': bool(image_path)
            }
        }
        
        # Add image-specific info
        if image_path:
            result['image_analysis'] = {
                'adaptive_risk_weight': float(adaptive_weight),
                'harmful_objects': {
                    'detected': bool(obj_features[0]),
                    'count': int(obj_features[1]),
                    'max_risk': float(obj_features[2]),
                    'avg_risk': float(obj_features[3]),
                    'weapon': bool(obj_features[4]),
                    'drugs': bool(obj_features[5]),
                    'alcohol': bool(obj_features[6])
                }
            }
        
        return result

# ============================================================================
# INTERACTIVE INTERFACE
# ============================================================================

def main():
    print("="*80)
    print("MULTI-MODAL SUICIDE DETECTION - INFERENCE")
    print("="*80)
    print()
    
    # Configuration
    FUSION_MODEL_PATH = "fusion_model/best_fusion_model.pth"
    TEXT_MODEL_PATH = "ckpts/checkpoint-4500"
    IMAGE_MODEL_PATH = "img_embed_new/weight_predictor.pth"
    OBJDEC_LABELS_DIR = "objdec/Harmful Object Detection/Labels"
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"Device: {DEVICE}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()
    
    # Initialize inference pipeline
    pipeline = SuicideDetectionInference(
        FUSION_MODEL_PATH, TEXT_MODEL_PATH, IMAGE_MODEL_PATH,
        OBJDEC_LABELS_DIR, DEVICE
    )
    
    print("="*80)
    print("READY FOR INFERENCE")
    print("="*80)
    print()
    print("Instructions:")
    print("  - Enter text when prompted (or press Enter to skip)")
    print("  - Enter image path when prompted (or press Enter to skip)")
    print("  - At least one input (text or image) is required")
    print("  - Type 'quit' to exit")
    print()
    print("="*80)
    
    while True:
        print("\n" + "-"*80)
        
        # Get text input
        text_input = input("Enter text: ").strip()
        if text_input.lower() == 'quit':
            print("\nExiting...")
            break
        
        # Get image path
        image_input = input("Enter image path (or press Enter to skip): ").strip()
        if image_input.lower() == 'quit':
            print("\nExiting...")
            break
        
        # Validate inputs
        if not text_input and not image_input:
            print("\n⚠️  Error: Please provide at least text or image input!")
            continue
        
        # Validate image path
        if image_input and not Path(image_input).exists():
            print(f"\n⚠️  Error: Image file not found: {image_input}")
            continue
        
        # Run prediction
        print("\n" + "="*80)
        print("ANALYZING...")
        print("="*80)
        
        try:
            result = pipeline.predict(
                text=text_input if text_input else None,
                image_path=image_input if image_input else None
            )
            
            # Display results
            print()
            if text_input:
                print(f"📝 TEXT: {text_input[:100]}{'...' if len(text_input) > 100 else ''}")
            if image_input:
                print(f"🖼️  IMAGE: {image_input}")
            print()
            
            print(f"🎯 PREDICTION: {result['prediction']}")
            print(f"   Confidence: {result['confidence']*100:.2f}%")
            print()
            
            print(f"📊 PROBABILITIES:")
            print(f"   Non-Suicide: {result['non_suicide_probability']*100:.2f}%")
            print(f"   Suicide:     {result['suicide_probability']*100:.2f}%")
            print()
            
            print(f"🔍 MODALITIES USED:")
            print(f"   Text: {'✓' if result['modalities_used']['text'] else '✗'}")
            print(f"   Image: {'✓' if result['modalities_used']['image'] else '✗'}")
            
            # Image analysis details
            if 'image_analysis' in result:
                print()
                print(f"🖼️  IMAGE ANALYSIS:")
                print(f"   Emotion Risk Weight: {result['image_analysis']['adaptive_risk_weight']:.3f}")
                
                harm_obj = result['image_analysis']['harmful_objects']
                if harm_obj['detected']:
                    print(f"   ⚠️  HARMFUL OBJECTS DETECTED:")
                    print(f"      Count: {harm_obj['count']}")
                    print(f"      Max Risk: {harm_obj['max_risk']:.2f}")
                    print(f"      Weapon: {'Yes' if harm_obj['weapon'] else 'No'}")
                    print(f"      Drugs: {'Yes' if harm_obj['drugs'] else 'No'}")
                    print(f"      Alcohol: {'Yes' if harm_obj['alcohol'] else 'No'}")
                else:
                    print(f"   No harmful objects detected")
            
            print()
            print("="*80)
            
        except Exception as e:
            print(f"\n✗ Error during prediction: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()
