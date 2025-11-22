"""
Multi-Modal Fusion Model for Suicide Detection
Handles: Text + Image Emotions + Harmful Object Detection
Supports optional inputs (text-only, image-only, or both)
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import json
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, 
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
    roc_auc_score
)
from PIL import Image
import torchvision.transforms as transforms
from collections import Counter
import os

# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# ============================================================================
# HARMFUL OBJECT DETECTOR
# ============================================================================

class HarmfulObjectDetector:
    """Detects harmful objects from label files"""
    
    HARMFUL_OBJECTS = {
        'knife': 0.9,
        'gun': 1.0,
        'rifle': 1.0,
        'pistol': 1.0,
        'weapon': 1.0,
        'blade': 0.85,
        'razor': 0.85,
        'pills': 0.7,
        'medication': 0.6,
        'drugs': 0.8,
        'syringe': 0.75,
        'needle': 0.7,
        'rope': 0.8,
        'noose': 1.0,
        'alcohol': 0.5,
        'cigarette': 0.3,
        'poison': 0.95,
        'gasoline': 0.7,
    }
    
    def __init__(self, labels_dir):
        self.labels_dir = Path(labels_dir)
    
    def get_label_file(self, image_path):
        """Get corresponding label file for an image"""
        image_name = Path(image_path).stem
        label_file = self.labels_dir / f"{image_name}.txt"
        return label_file if label_file.exists() else None
    
    def extract_features(self, image_path):
        """
        Extract harmful object features from label file
        Returns: dict with detection flags and risk score
        """
        label_file = self.get_label_file(image_path)
        
        features = {
            'has_harmful_object': 0,
            'harmful_object_count': 0,
            'max_risk_score': 0.0,
            'avg_risk_score': 0.0,
            'weapon_detected': 0,
            'drug_detected': 0,
            'alcohol_detected': 0,
        }
        
        if label_file is None:
            return features
        
        try:
            with open(label_file, 'r') as f:
                objects = [line.strip().lower() for line in f if line.strip()]
            
            if not objects:
                return features
            
            risk_scores = []
            for obj in objects:
                # Check if object is harmful
                for harmful_obj, risk in self.HARMFUL_OBJECTS.items():
                    if harmful_obj in obj:
                        features['has_harmful_object'] = 1
                        features['harmful_object_count'] += 1
                        risk_scores.append(risk)
                        
                        # Specific categories
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
            print(f"Error reading {label_file}: {e}")
        
        return features

# ============================================================================
# MULTI-MODAL DATASET
# ============================================================================

class MultiModalDataset(Dataset):
    """Dataset that handles text, image, and object detection features"""
    
    def __init__(self, data, text_embeddings=None, image_embeddings=None, 
                 harmful_object_detector=None, mode='train'):
        """
        Args:
            data: List of dicts with 'text', 'image_path', 'label'
            text_embeddings: Pre-computed text embeddings (optional)
            image_embeddings: Pre-computed image embeddings (optional)
            harmful_object_detector: HarmfulObjectDetector instance (optional)
            mode: 'train' or 'test'
        """
        self.data = data
        self.text_embeddings = text_embeddings
        self.image_embeddings = image_embeddings
        self.harmful_object_detector = harmful_object_detector
        self.mode = mode
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Text embedding (768-dim) - use zeros if not available
        if self.text_embeddings is not None and item.get('text_idx') is not None:
            text_emb = self.text_embeddings[item['text_idx']]
        else:
            text_emb = np.zeros(768, dtype=np.float32)
        
        # Image embedding (1280-dim) - use zeros if not available
        if self.image_embeddings is not None and item.get('image_idx') is not None:
            image_emb = self.image_embeddings[item['image_idx']]
        else:
            image_emb = np.zeros(1280, dtype=np.float32)
        
        # Harmful object features (7-dim)
        if self.harmful_object_detector is not None and item.get('image_path'):
            obj_features = self.harmful_object_detector.extract_features(item['image_path'])
            obj_emb = np.array([
                obj_features['has_harmful_object'],
                obj_features['harmful_object_count'],
                obj_features['max_risk_score'],
                obj_features['avg_risk_score'],
                obj_features['weapon_detected'],
                obj_features['drug_detected'],
                obj_features['alcohol_detected'],
            ], dtype=np.float32)
        else:
            obj_emb = np.zeros(7, dtype=np.float32)
        
        # Label
        label = item['label']
        
        # Modality flags (which inputs are present)
        has_text = 1 if item.get('text_idx') is not None else 0
        has_image = 1 if item.get('image_idx') is not None else 0
        
        return {
            'text_emb': torch.tensor(text_emb, dtype=torch.float32),
            'image_emb': torch.tensor(image_emb, dtype=torch.float32),
            'obj_emb': torch.tensor(obj_emb, dtype=torch.float32),
            'has_text': torch.tensor(has_text, dtype=torch.float32),
            'has_image': torch.tensor(has_image, dtype=torch.float32),
            'label': torch.tensor(label, dtype=torch.long)
        }

# ============================================================================
# FUSION MODEL
# ============================================================================

class MultiModalFusionModel(nn.Module):
    """
    Fusion model combining:
    - Text embeddings (768-dim)
    - Image emotion embeddings (1280-dim)
    - Harmful object features (7-dim)
    Total input: 2055-dim
    """
    
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
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using Xavier uniform"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, text_emb, image_emb, obj_emb, has_text, has_image):
        """
        Args:
            text_emb: (batch_size, 768)
            image_emb: (batch_size, 1280)
            obj_emb: (batch_size, 7)
            has_text: (batch_size,) - binary flag
            has_image: (batch_size,) - binary flag
        """
        batch_size = text_emb.size(0)
        
        # Project each modality
        text_features = self.text_projection(text_emb)
        image_features = self.image_projection(image_emb)
        obj_features = self.obj_projection(obj_emb)
        
        # Concatenate all features + modality flags
        modality_flags = torch.stack([has_text, has_image], dim=1)  # (batch_size, 2)
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
# TRAINING FUNCTIONS
# ============================================================================

def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    for batch in tqdm(dataloader, desc="Training"):
        # Move to device
        text_emb = batch['text_emb'].to(device)
        image_emb = batch['image_emb'].to(device)
        obj_emb = batch['obj_emb'].to(device)
        has_text = batch['has_text'].to(device)
        has_image = batch['has_image'].to(device)
        labels = batch['label'].to(device)
        
        # Forward pass
        optimizer.zero_grad()
        logits = model(text_emb, image_emb, obj_emb, has_text, has_image)
        loss = criterion(logits, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Track metrics
        total_loss += loss.item()
        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    
    return avg_loss, accuracy

def evaluate(model, dataloader, criterion, device):
    """Evaluate model"""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            # Move to device
            text_emb = batch['text_emb'].to(device)
            image_emb = batch['image_emb'].to(device)
            obj_emb = batch['obj_emb'].to(device)
            has_text = batch['has_text'].to(device)
            has_image = batch['has_image'].to(device)
            labels = batch['label'].to(device)
            
            # Forward pass
            logits = model(text_emb, image_emb, obj_emb, has_text, has_image)
            loss = criterion(logits, labels)
            
            # Track metrics
            total_loss += loss.item()
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='binary'
    )
    
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except:
        auc = 0.0
    
    return {
        'loss': avg_loss,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc
    }

# ============================================================================
# MAIN TRAINING
# ============================================================================

def main():
    print("="*80)
    print("MULTI-MODAL FUSION MODEL TRAINING")
    print("="*80)
    print()
    
    # Configuration
    TEXT_EMBED_DIR = Path("text_embed")
    IMAGE_EMBED_DIR = Path("img_embed_new")  # Using adaptive embeddings
    OBJDEC_LABELS_DIR = Path("objdec/Harmful Object Detection/Labels")
    OUTPUT_DIR = Path("fusion_model")
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    BATCH_SIZE = 64
    NUM_EPOCHS = 20
    LEARNING_RATE = 0.001
    WEIGHT_DECAY = 0.01
    
    # Load text embeddings
    print("Loading text embeddings...")
    text_data = np.load(TEXT_EMBED_DIR / "text_embeddings.npz", allow_pickle=True)
    text_embeddings = text_data['embeddings']  # (218875, 768)
    text_labels = text_data['labels']
    text_texts = text_data['texts']
    print(f"✓ Loaded {len(text_embeddings):,} text embeddings")
    
    # Load image embeddings
    print("Loading image embeddings...")
    train_img_data = np.load(IMAGE_EMBED_DIR / "train_embeddings.npz", allow_pickle=True)
    test_img_data = np.load(IMAGE_EMBED_DIR / "test_embeddings.npz", allow_pickle=True)
    
    train_image_embeddings = train_img_data['embeddings']  # (N, 1280)
    train_image_emotions = train_img_data['emotions']
    train_image_paths = train_img_data['img_paths']
    
    test_image_embeddings = test_img_data['embeddings']
    test_image_emotions = test_img_data['emotions']
    test_image_paths = test_img_data['img_paths']
    
    # Create labels from emotions: suicide emotions = 1, non-suicide = 0
    # Suicide emotions: sad, fearful, angry, disgusted
    # Non-suicide emotions: happy, neutral, surprised
    suicide_emotions = {'sad', 'fearful', 'angry', 'disgusted'}
    non_suicide_emotions = {'happy', 'neutral', 'surprised'}
    
    train_image_labels = np.array([
        1 if emotion in suicide_emotions else 0 
        for emotion in train_image_emotions
    ])
    test_image_labels = np.array([
        1 if emotion in suicide_emotions else 0 
        for emotion in test_image_emotions
    ])
    
    # Combine train and test images
    all_image_embeddings = np.concatenate([train_image_embeddings, test_image_embeddings])
    all_image_labels = np.concatenate([train_image_labels, test_image_labels])
    all_image_paths = np.concatenate([train_image_paths, test_image_paths])
    
    print(f"✓ Loaded {len(all_image_embeddings):,} image embeddings")
    
    # Initialize harmful object detector
    print("Initializing harmful object detector...")
    obj_detector = HarmfulObjectDetector(OBJDEC_LABELS_DIR)
    print(f"✓ Object detector ready ({len(obj_detector.HARMFUL_OBJECTS)} harmful objects)")
    
    # Create multi-modal dataset
    print("\nCreating multi-modal dataset...")
    
    # Strategy: Sample balanced data from text and image datasets
    # Take equal numbers from each modality
    
    n_samples_per_modality = min(len(text_embeddings), len(all_image_embeddings)) // 2
    
    # Text-only samples
    text_only_data = []
    for i in range(min(n_samples_per_modality, len(text_embeddings))):
        text_only_data.append({
            'text_idx': i,
            'image_idx': None,
            'image_path': None,
            'label': int(text_labels[i])
        })
    
    # Image-only samples  
    image_only_data = []
    for i in range(min(n_samples_per_modality, len(all_image_embeddings))):
        image_only_data.append({
            'text_idx': None,
            'image_idx': i,
            'image_path': all_image_paths[i],
            'label': int(all_image_labels[i])
        })
    
    # Combined samples (randomly pair text and images with same label)
    combined_data = []
    suicide_text_indices = np.where(text_labels == 1)[0]
    non_suicide_text_indices = np.where(text_labels == 0)[0]
    suicide_image_indices = np.where(all_image_labels == 1)[0]
    non_suicide_image_indices = np.where(all_image_labels == 0)[0]
    
    # Sample combined pairs
    n_combined = min(
        len(suicide_text_indices), len(suicide_image_indices),
        len(non_suicide_text_indices), len(non_suicide_image_indices)
    ) // 2
    
    for i in range(n_combined):
        # Suicide pair
        combined_data.append({
            'text_idx': int(suicide_text_indices[i]),
            'image_idx': int(suicide_image_indices[i]),
            'image_path': all_image_paths[suicide_image_indices[i]],
            'label': 1
        })
        # Non-suicide pair
        combined_data.append({
            'text_idx': int(non_suicide_text_indices[i]),
            'image_idx': int(non_suicide_image_indices[i]),
            'image_path': all_image_paths[non_suicide_image_indices[i]],
            'label': 0
        })
    
    # Combine all data
    all_data = text_only_data + image_only_data + combined_data
    np.random.shuffle(all_data)
    
    print(f"✓ Created dataset:")
    print(f"  Text-only: {len(text_only_data):,}")
    print(f"  Image-only: {len(image_only_data):,}")
    print(f"  Combined: {len(combined_data):,}")
    print(f"  Total: {len(all_data):,}")
    
    label_counts = Counter([d['label'] for d in all_data])
    print(f"  Suicide: {label_counts[1]:,}")
    print(f"  Non-suicide: {label_counts[0]:,}")
    
    # Train/val split
    train_data, val_data = train_test_split(all_data, test_size=0.15, random_state=42, 
                                            stratify=[d['label'] for d in all_data])
    
    print(f"\nTrain: {len(train_data):,} | Val: {len(val_data):,}")
    
    # Create datasets
    train_dataset = MultiModalDataset(
        train_data, text_embeddings, all_image_embeddings, obj_detector, mode='train'
    )
    val_dataset = MultiModalDataset(
        val_data, text_embeddings, all_image_embeddings, obj_detector, mode='val'
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=0, pin_memory=True if torch.cuda.is_available() else False
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=0, pin_memory=True if torch.cuda.is_available() else False
    )
    
    # Initialize model
    print("\nInitializing fusion model...")
    model = MultiModalFusionModel(
        text_dim=768,
        image_dim=1280,
        obj_dim=7,
        hidden_dim=512,
        dropout=0.3
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"✓ Model created with {total_params:,} parameters")
    
    # Loss and optimizer
    # Use class weights to handle imbalance
    label_counts = Counter([d['label'] for d in train_data])
    class_weights = torch.tensor([
        1.0 / label_counts[0],
        1.0 / label_counts[1]
    ], dtype=torch.float32).to(device)
    class_weights = class_weights / class_weights.sum()  # Normalize
    
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3
    )
    
    # Training loop
    print("\n" + "="*80)
    print("STARTING TRAINING")
    print("="*80)
    
    best_f1 = 0
    best_epoch = 0
    training_history = []
    
    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")
        print("-" * 80)
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Evaluate
        val_metrics = evaluate(model, val_loader, criterion, device)
        
        # Scheduler step
        scheduler.step(val_metrics['f1'])
        
        # Print metrics
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_metrics['loss']:.4f} | Val Acc: {val_metrics['accuracy']:.4f}")
        print(f"Val Precision: {val_metrics['precision']:.4f} | Val Recall: {val_metrics['recall']:.4f}")
        print(f"Val F1: {val_metrics['f1']:.4f} | Val AUC: {val_metrics['auc']:.4f}")
        
        # Save history
        training_history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_metrics['loss'],
            'val_acc': val_metrics['accuracy'],
            'val_precision': val_metrics['precision'],
            'val_recall': val_metrics['recall'],
            'val_f1': val_metrics['f1'],
            'val_auc': val_metrics['auc']
        })
        
        # Save best model
        if val_metrics['f1'] > best_f1:
            best_f1 = val_metrics['f1']
            best_epoch = epoch + 1
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_f1': best_f1,
                'val_metrics': val_metrics
            }, OUTPUT_DIR / 'best_fusion_model.pth')
            print(f"✓ Saved best model (F1: {best_f1:.4f})")
    
    print("\n" + "="*80)
    print("TRAINING COMPLETED")
    print("="*80)
    print(f"Best F1: {best_f1:.4f} at epoch {best_epoch}")
    
    # Save training history
    with open(OUTPUT_DIR / 'training_history.json', 'w') as f:
        json.dump(training_history, f, indent=2)
    
    print(f"\n✓ Training history saved to {OUTPUT_DIR / 'training_history.json'}")
    print(f"✓ Best model saved to {OUTPUT_DIR / 'best_fusion_model.pth'}")

if __name__ == "__main__":
    main()
