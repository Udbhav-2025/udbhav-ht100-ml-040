import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import json
from tqdm import tqdm
from pathlib import Path
from sklearn.model_selection import train_test_split

# Set random seeds for reproducibility
def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

# Initial Emotion to Suicide Risk Mapping (Hardcoded baseline)
INITIAL_EMOTION_RISK_MAPPING = {
    'angry': {
        'risk_level': 'high',
        'initial_suicide_weight': 0.85,
        'initial_non_suicide_weight': 0.15,
        'description': 'Anger can indicate aggression and suicidal ideation'
    },
    'disgusted': {
        'risk_level': 'high',
        'initial_suicide_weight': 0.80,
        'initial_non_suicide_weight': 0.20,
        'description': 'Self-disgust is a strong suicide risk factor'
    },
    'fearful': {
        'risk_level': 'high',
        'initial_suicide_weight': 0.90,
        'initial_non_suicide_weight': 0.10,
        'description': 'Fear and anxiety strongly associated with suicide risk'
    },
    'sad': {
        'risk_level': 'high',
        'initial_suicide_weight': 0.95,
        'initial_non_suicide_weight': 0.05,
        'description': 'Sadness/depression is primary suicide risk indicator'
    },
    'happy': {
        'risk_level': 'low',
        'initial_suicide_weight': 0.10,
        'initial_non_suicide_weight': 0.90,
        'description': 'Positive emotion indicates lower suicide risk'
    },
    'neutral': {
        'risk_level': 'low',
        'initial_suicide_weight': 0.30,
        'initial_non_suicide_weight': 0.70,
        'description': 'Neutral emotion with slight risk (emotional numbness)'
    },
    'surprised': {
        'risk_level': 'ambiguous',
        'initial_suicide_weight': 0.50,
        'initial_non_suicide_weight': 0.50,
        'description': 'Surprise can be positive or negative context'
    }
}

class EmotionImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.images = []
        self.labels = []
        self.emotion_names = []
        
        # Load all images from emotion folders
        for emotion_folder in self.root_dir.iterdir():
            if emotion_folder.is_dir():
                emotion = emotion_folder.name
                for img_path in emotion_folder.glob('*'):
                    if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                        self.images.append(str(img_path))
                        self.emotion_names.append(emotion)
                        # Store initial risk weights
                        self.labels.append(INITIAL_EMOTION_RISK_MAPPING.get(emotion, {
                            'initial_suicide_weight': 0.5,
                            'initial_non_suicide_weight': 0.5
                        }))
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        emotion = self.emotion_names[idx]
        risk_info = self.labels[idx]
        
        # Load image
        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            image = torch.zeros(3, 224, 224)
        
        return {
            'image': image,
            'emotion': emotion,
            'initial_suicide_weight': risk_info['initial_suicide_weight'],
            'initial_non_suicide_weight': risk_info['initial_non_suicide_weight'],
            'risk_level': risk_info['risk_level'],
            'img_path': img_path
        }

class EfficientNetEmbedder(nn.Module):
    def __init__(self):
        super(EfficientNetEmbedder, self).__init__()
        # Load pre-trained EfficientNet-B0
        self.efficientnet = models.efficientnet_b0(weights='DEFAULT')
        
        # Remove the classification head to get embeddings
        self.efficientnet.classifier = nn.Identity()
        self.embedding_dim = 1280
        
    def forward(self, x):
        return self.efficientnet(x)

class SuicideRiskWeightPredictor(nn.Module):
    """
    Neural network that learns to adjust suicide risk weights based on image features
    Takes EfficientNet embeddings + initial weights → predicts adjusted weights
    """
    def __init__(self, embedding_dim=1280, hidden_dim=512):
        super(SuicideRiskWeightPredictor, self).__init__()
        
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
        
        # Output: adjusted suicide weight (single value between 0-1)
        self.fc_out = nn.Linear(128, 1)
        
    def forward(self, embeddings, initial_weights):
        # Ensure initial_weights is the right shape [batch_size, 1]
        if initial_weights.dim() == 1:
            initial_weights = initial_weights.unsqueeze(1)
        
        # Concatenate embeddings with initial weights
        x = torch.cat([embeddings, initial_weights], dim=1)
        
        # Forward pass through network
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout1(x)
        
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout2(x)
        
        x = F.relu(self.bn3(self.fc3(x)))
        x = self.dropout3(x)
        
        # Output: adjusted weight (sigmoid to keep in [0, 1])
        adjusted_weight = torch.sigmoid(self.fc_out(x))
        
        return adjusted_weight.squeeze()

def train_weight_predictor(embeddings, initial_weights, emotions, device, epochs=50, batch_size=128):
    """
    Train a neural network to refine suicide risk weights based on image features
    """
    print("\n" + "="*80)
    print("TRAINING WEIGHT PREDICTOR")
    print("="*80 + "\n")
    
    # Convert to tensors
    X_embed = torch.FloatTensor(embeddings).to(device)
    X_init_weights = torch.FloatTensor(initial_weights).to(device)
    
    # Create targets: we'll use initial weights as soft targets
    # but allow the model to learn adjustments based on visual features
    y_targets = torch.FloatTensor(initial_weights).to(device)
    
    # Split data
    indices = np.arange(len(embeddings))
    train_idx, val_idx = train_test_split(indices, test_size=0.2, random_state=42, stratify=emotions)
    
    # Initialize model
    model = SuicideRiskWeightPredictor(embedding_dim=1280).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    criterion = nn.MSELoss()
    
    print(f"Training samples: {len(train_idx)}")
    print(f"Validation samples: {len(val_idx)}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print()
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_losses = []
        
        # Mini-batch training
        for i in range(0, len(train_idx), batch_size):
            batch_idx = train_idx[i:i+batch_size]
            
            batch_embed = X_embed[batch_idx].float()  # Ensure float32
            batch_init_weights = X_init_weights[batch_idx].float().unsqueeze(1)  # Ensure float32 and correct shape
            batch_targets = y_targets[batch_idx].float().unsqueeze(1)  # Ensure float32 and correct shape
            
            optimizer.zero_grad()
            predictions = model(batch_embed, batch_init_weights)
            loss = criterion(predictions, batch_targets)
            
            loss.backward()
            optimizer.step()
            
            train_losses.append(loss.item())
        
        # Validation
        model.eval()
        val_losses = []
        with torch.no_grad():
            for i in range(0, len(val_idx), batch_size):
                batch_idx = val_idx[i:i+batch_size]
                
                batch_embed = X_embed[batch_idx].float()  # Ensure float32
                batch_init_weights = X_init_weights[batch_idx].float().unsqueeze(1)  # Ensure float32 and correct shape
                batch_targets = y_targets[batch_idx].float().unsqueeze(1)  # Ensure float32 and correct shape
                
                predictions = model(batch_embed, batch_init_weights)
                loss = criterion(predictions, batch_targets)
                val_losses.append(loss.item())
        
        avg_train_loss = np.mean(train_losses)
        avg_val_loss = np.mean(val_losses)
        
        scheduler.step(avg_val_loss)
        
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")
        
        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            # Save best model
            best_model_state = model.state_dict()
        else:
            patience_counter += 1
            if patience_counter >= 10:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break
    
    # Load best model
    model.load_state_dict(best_model_state)
    print(f"\n✓ Training complete! Best validation loss: {best_val_loss:.6f}\n")
    
    return model

def create_embeddings_with_learned_weights(data_dir, output_dir, batch_size=32, num_workers=0):
    """
    Generate embeddings with dynamically adjusted suicide risk weights
    """
    print("="*80)
    print("ADAPTIVE SUICIDE RISK WEIGHT GENERATION WITH EFFICIENTNET-B0")
    print("="*80)
    print()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    print()
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Image transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Load EfficientNet embedder
    print("Loading EfficientNet-B0 model...")
    embedder = EfficientNetEmbedder().to(device)
    embedder.eval()
    print(f"✓ Model loaded: EfficientNet-B0")
    print(f"✓ Embedding dimension: {embedder.embedding_dim}\n")
    
    # First pass: Extract embeddings from training set
    train_dir = Path(data_dir) / 'train'
    if not train_dir.exists():
        raise RuntimeError(f"Training directory not found: {train_dir}")
    
    print("="*80)
    print("PHASE 1: EXTRACTING EMBEDDINGS FROM TRAINING SET")
    print("="*80 + "\n")
    
    train_dataset = EmotionImageDataset(train_dir, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, 
                             num_workers=num_workers, pin_memory=torch.cuda.is_available())
    
    print(f"Total training images: {len(train_dataset)}\n")
    
    train_embeddings_list = []
    train_emotions_list = []
    train_initial_weights_list = []
    train_risk_levels_list = []
    train_paths_list = []
    
    print("Extracting embeddings...")
    with torch.no_grad():
        for batch in tqdm(train_loader, desc="Processing training images"):
            images = batch['image'].to(device)
            embeddings = embedder(images)
            
            train_embeddings_list.append(embeddings.cpu().numpy())
            train_emotions_list.extend(batch['emotion'])
            train_initial_weights_list.extend(batch['initial_suicide_weight'].numpy())
            train_risk_levels_list.extend(batch['risk_level'])
            train_paths_list.extend(batch['img_path'])
            
            if torch.cuda.is_available() and len(train_embeddings_list) % 10 == 0:
                torch.cuda.empty_cache()
    
    train_embeddings = np.vstack(train_embeddings_list)
    train_initial_weights = np.array(train_initial_weights_list)
    
    print(f"\n✓ Extracted {len(train_embeddings)} training embeddings\n")
    
    # Phase 2: Train weight predictor
    print("="*80)
    print("PHASE 2: TRAINING ADAPTIVE WEIGHT PREDICTOR")
    print("="*80)
    
    weight_predictor = train_weight_predictor(
        train_embeddings, 
        train_initial_weights,
        train_emotions_list,
        device,
        epochs=50,
        batch_size=128
    )
    
    # Phase 3: Generate adjusted weights for all data
    print("="*80)
    print("PHASE 3: GENERATING ADJUSTED WEIGHTS FOR ALL IMAGES")
    print("="*80 + "\n")
    
    for split in ['train', 'test']:
        split_dir = Path(data_dir) / split
        if not split_dir.exists():
            print(f"⚠ Skipping {split} - directory not found\n")
            continue
        
        print(f"\nProcessing {split.upper()} set...")
        print("-"*80)
        
        dataset = EmotionImageDataset(split_dir, transform=transform)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=torch.cuda.is_available())
        
        all_embeddings = []
        all_emotions = []
        all_initial_weights = []
        all_adjusted_weights = []
        all_risk_levels = []
        all_img_paths = []
        
        embedder.eval()
        weight_predictor.eval()
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc=f"Processing {split}"):
                images = batch['image'].to(device)
                initial_weights = batch['initial_suicide_weight'].float().to(device)
                
                # Get embeddings
                embeddings = embedder(images).float()  # Ensure float32
                
                # Predict adjusted weights (forward method handles unsqueeze)
                adjusted_weights = weight_predictor(embeddings, initial_weights)
                
                all_embeddings.append(embeddings.cpu().numpy())
                all_emotions.extend(batch['emotion'])
                all_initial_weights.extend(batch['initial_suicide_weight'].numpy())
                all_adjusted_weights.extend(adjusted_weights.cpu().numpy())
                all_risk_levels.extend(batch['risk_level'])
                all_img_paths.extend(batch['img_path'])
        
        all_embeddings = np.vstack(all_embeddings)
        all_initial_weights = np.array(all_initial_weights)
        all_adjusted_weights = np.array(all_adjusted_weights)
        
        # Calculate adjustment statistics
        weight_changes = all_adjusted_weights - all_initial_weights
        
        print(f"\n✓ Generated {len(all_embeddings)} embeddings with adjusted weights")
        print(f"\nWeight Adjustment Statistics:")
        print(f"  Mean change: {np.mean(weight_changes):+.4f}")
        print(f"  Std change:  {np.std(weight_changes):.4f}")
        print(f"  Max increase: {np.max(weight_changes):+.4f}")
        print(f"  Max decrease: {np.min(weight_changes):+.4f}")
        
        # Show per-emotion statistics
        print(f"\nPer-Emotion Weight Adjustments:")
        for emotion in sorted(set(all_emotions)):
            mask = np.array(all_emotions) == emotion
            initial = all_initial_weights[mask]
            adjusted = all_adjusted_weights[mask]
            change = adjusted - initial
            
            print(f"  {emotion:12s}: Initial={np.mean(initial):.3f} → "
                  f"Adjusted={np.mean(adjusted):.3f} (Δ={np.mean(change):+.3f})")
        
        # Save embeddings
        output_file = output_dir / f"{split}_embeddings.npz"
        print(f"\nSaving to: {output_file}")
        
        np.savez_compressed(
            output_file,
            embeddings=all_embeddings,
            emotions=np.array(all_emotions),
            initial_suicide_weights=all_initial_weights,
            adjusted_suicide_weights=all_adjusted_weights,
            adjusted_non_suicide_weights=1.0 - all_adjusted_weights,
            risk_levels=np.array(all_risk_levels),
            img_paths=np.array(all_img_paths)
        )
        
        # Save metadata
        metadata_file = output_dir / f"{split}_metadata.json"
        metadata = {
            'split': split,
            'total_images': len(all_embeddings),
            'embedding_dim': embedder.embedding_dim,
            'model': 'EfficientNet-B0 + Adaptive Weight Predictor',
            'approach': 'Hardcoded baseline + Neural adjustment',
            'initial_risk_mapping': INITIAL_EMOTION_RISK_MAPPING,
            'device': str(device)
        }
        
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"✓ {split.upper()} set complete!\n")
    
    # Save weight predictor model
    model_path = output_dir / "weight_predictor.pth"
    torch.save(weight_predictor.state_dict(), model_path)
    print(f"✓ Weight predictor model saved to: {model_path}\n")
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    print("="*80)
    print("ADAPTIVE EMBEDDING GENERATION COMPLETE!")
    print("="*80)
    print(f"\nAll files saved to: {output_dir}")
    print("\nApproach: Started with hardcoded weights, adjusted by neural network")
    print("based on visual features learned from EfficientNet-B0 embeddings.\n")

if __name__ == "__main__":
    DATA_DIR = "emotion_images"
    OUTPUT_DIR = "img_embed_new"
    BATCH_SIZE = 32
    NUM_WORKERS = 0
    
    create_embeddings_with_learned_weights(
        data_dir=DATA_DIR,
        output_dir=OUTPUT_DIR,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS
    )
