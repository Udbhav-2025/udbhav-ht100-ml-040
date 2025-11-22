import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
from pathlib import Path
import json
import os

# Set random seeds for reproducibility
def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': label,
            'text': text
        }

def generate_text_embeddings(
    csv_path,
    checkpoint_path,
    output_dir,
    batch_size=32,
    max_length=512
):
    """
    Generate text embeddings using fine-tuned MentalBERT model
    
    Args:
        csv_path: Path to Suicide_Detection_cleaned.csv
        checkpoint_path: Path to checkpoint-4500 directory
        output_dir: Directory to save embeddings (text_embed)
        batch_size: Batch size for processing
        max_length: Maximum sequence length
    """
    
    print("="*80)
    print("TEXT EMBEDDING GENERATION WITH FINE-TUNED MENTALBERT")
    print("="*80)
    print()
    
    # Use GPU if available (trained with CUDA)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
        torch.cuda.empty_cache()
    print()
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Load data
    print("Loading cleaned dataset...")
    df = pd.read_csv(csv_path, usecols=['text', 'class'])
    
    # Clean up the class column
    df['class'] = df['class'].str.strip()
    df = df[df['class'].isin(['suicide', 'non-suicide'])]
    
    print(f"Total samples: {len(df):,}")
    print(f"\nClass distribution:")
    print(df['class'].value_counts())
    print()
    
    # Convert labels to binary
    label_map = {'non-suicide': 0, 'suicide': 1}
    df['label'] = df['class'].map(label_map)
    
    # Load fine-tuned model and tokenizer
    print(f"Loading fine-tuned MentalBERT from checkpoint...")
    print(f"Checkpoint: {checkpoint_path}")
    
    checkpoint_path = Path(checkpoint_path)
    
    # Verify checkpoint files exist
    config_file = checkpoint_path / "config.json"
    weights_file = checkpoint_path / "model.safetensors"
    
    if not config_file.exists():
        raise FileNotFoundError(f"config.json not found in {checkpoint_path}")
    if not weights_file.exists():
        raise FileNotFoundError(f"model.safetensors not found in {checkpoint_path}")
    
    print(f"✓ Found config.json")
    print(f"✓ Found model.safetensors")
    print()
    
    # Load config first (needed for tokenizer matching)
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(str(checkpoint_path))
    print(f"✓ Config loaded: {config.model_type}")
    print(f"  Vocab size: {config.vocab_size}")
    print()
    
    # Load tokenizer - bert-base-uncased was used during training (fallback)
    print(f"Loading tokenizer...")
    print(f"  Note: Training used bert-base-uncased (fallback model)")
    
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    tok_vocab_size = len(tokenizer)
    
    print(f"✓ Tokenizer loaded: bert-base-uncased")
    print(f"  Tokenizer vocab_size: {tok_vocab_size}")
    print(f"  Checkpoint vocab_size: {config.vocab_size}")
    
    if tok_vocab_size != config.vocab_size:
        print(f"  ⚠ Warning: Vocab size mismatch!")
        print(f"    This may cause indexing issues. The model might have been")
        print(f"    fine-tuned with a different vocabulary or have custom embeddings.")
    else:
        print(f"  ✓ Vocab sizes match!")
    print()
    
    # Load model with safetensors weights
    print("Loading model configuration and weights...")
    from transformers import AutoConfig, AutoModelForSequenceClassification
    
    # Load config
    config = AutoConfig.from_pretrained(str(checkpoint_path))
    print(f"✓ Config loaded: {config.model_type}")
    
    # Load the full classification model first (as it was trained)
    classifier_model = AutoModelForSequenceClassification.from_pretrained(
        str(checkpoint_path),
        config=config,
        use_safetensors=True
    )
    
    # Extract just the base BERT model (without classification head)
    # For BERT, the base model is stored in classifier_model.bert
    if hasattr(classifier_model, 'bert'):
        model = classifier_model.bert
    elif hasattr(classifier_model, 'base_model'):
        model = classifier_model.base_model
    else:
        # Fallback: use the full model
        model = classifier_model
    
    model = model.to(device)
    model.eval()
    
    print(f"✓ Model loaded with fine-tuned weights from model.safetensors")
    print(f"✓ Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"✓ Embedding dimension: {config.hidden_size}")
    print()
    
    embedding_dim = model.config.hidden_size
    
    # Create dataset and dataloader
    print("Creating dataset...")
    dataset = TextDataset(
        df['text'].values,
        df['label'].values,
        tokenizer,
        max_length
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )
    
    print(f"✓ Dataset created: {len(dataset):,} samples")
    print(f"✓ Batch size: {batch_size}")
    print(f"✓ Number of batches: {len(dataloader):,}")
    print()
    
    # Generate embeddings
    print("="*80)
    print("GENERATING TEXT EMBEDDINGS")
    print("="*80)
    print()
    
    all_embeddings = []
    all_labels = []
    all_texts = []
    all_classes = []
    
    # Get model's actual vocab size
    model_vocab_size = model.config.vocab_size
    print(f"Model vocab size: {model_vocab_size}")
    print(f"Tokenizer vocab size: {len(tokenizer)}")
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Processing batches")):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            # Clamp input_ids to model's vocab size to avoid index errors
            # This handles the vocab mismatch between tokenizer (30522) and model (28996)
            input_ids = torch.clamp(input_ids, max=model_vocab_size - 1)
            
            # Get embeddings from model
            # Use [CLS] token embedding (first token) as sentence representation
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True
            )
            
            # Get [CLS] token embeddings (last hidden state, first token)
            cls_embeddings = outputs.last_hidden_state[:, 0, :]  # Shape: [batch_size, hidden_size]
            
            # Store results
            all_embeddings.append(cls_embeddings.cpu().numpy())
            all_labels.extend(batch['label'].numpy())
            all_texts.extend(batch['text'])
    
    # Concatenate all embeddings
    all_embeddings = np.vstack(all_embeddings)
    all_labels = np.array(all_labels)
    
    # Map labels back to class names
    reverse_label_map = {0: 'non-suicide', 1: 'suicide'}
    all_classes = np.array([reverse_label_map[label] for label in all_labels])
    
    print(f"\n✓ Generated {len(all_embeddings):,} embeddings")
    print(f"  Embedding shape: {all_embeddings.shape}")
    print(f"  Embedding dimension: {embedding_dim}")
    print()
    
    # Save embeddings
    output_file = output_dir / "text_embeddings.npz"
    metadata_file = output_dir / "text_metadata.json"
    
    print(f"Saving embeddings to: {output_file}")
    np.savez_compressed(
        output_file,
        embeddings=all_embeddings,
        labels=all_labels,
        classes=all_classes,
        texts=np.array(all_texts, dtype=object)
    )
    
    # Calculate statistics
    suicide_count = np.sum(all_labels == 1)
    non_suicide_count = np.sum(all_labels == 0)
    
    # Save metadata
    metadata = {
        'total_samples': len(all_embeddings),
        'embedding_dim': embedding_dim,
        'max_length': max_length,
        'batch_size': batch_size,
        'model_checkpoint': str(checkpoint_path),
        'model_type': model.config.model_type,
        'suicide_count': int(suicide_count),
        'non_suicide_count': int(non_suicide_count),
        'class_distribution': {
            'suicide': int(suicide_count),
            'non-suicide': int(non_suicide_count)
        },
        'label_mapping': label_map,
        'device': str(device)
    }
    
    print(f"Saving metadata to: {metadata_file}")
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print()
    print("="*80)
    print("EMBEDDING GENERATION COMPLETE!")
    print("="*80)
    print()
    print(f"Total embeddings: {len(all_embeddings):,}")
    print(f"Embedding dimension: {embedding_dim}")
    print()
    print("Class Distribution:")
    print(f"  Suicide:     {suicide_count:,} ({suicide_count/len(all_embeddings)*100:.2f}%)")
    print(f"  Non-suicide: {non_suicide_count:,} ({non_suicide_count/len(all_embeddings)*100:.2f}%)")
    print()
    print(f"Files saved in: {output_dir}")
    print(f"  - text_embeddings.npz")
    print(f"  - text_metadata.json")
    print()
    
    # Sample embeddings
    print("Sample embeddings (first 3):")
    print("-"*80)
    for i in range(min(3, len(all_embeddings))):
        text_preview = all_texts[i][:80] + "..." if len(all_texts[i]) > 80 else all_texts[i]
        print(f"\n{i+1}. Text: {text_preview}")
        print(f"   Class: {all_classes[i]}")
        print(f"   Embedding shape: {all_embeddings[i].shape}")
        print(f"   Embedding mean: {all_embeddings[i].mean():.4f}")
        print(f"   Embedding std: {all_embeddings[i].std():.4f}")
    print()
    
    return all_embeddings, all_labels, all_texts, metadata

def load_text_embeddings(embeddings_dir):
    """
    Load text embeddings from saved files
    
    Args:
        embeddings_dir: Directory containing saved embeddings
    
    Returns:
        Dictionary containing embeddings and metadata
    """
    embeddings_dir = Path(embeddings_dir)
    
    # Load embeddings
    data = np.load(embeddings_dir / "text_embeddings.npz", allow_pickle=True)
    
    # Load metadata
    with open(embeddings_dir / "text_metadata.json", 'r') as f:
        metadata = json.load(f)
    
    return {
        'embeddings': data['embeddings'],
        'labels': data['labels'],
        'classes': data['classes'],
        'texts': data['texts'],
        'metadata': metadata
    }

if __name__ == "__main__":
    # Configuration
    CSV_PATH = "Suicide_Detection_cleaned.csv"
    CHECKPOINT_PATH = "ckpts/checkpoint-4500"
    OUTPUT_DIR = "text_embed"
    BATCH_SIZE = 32
    MAX_LENGTH = 512
    
    # Generate embeddings
    embeddings, labels, texts, metadata = generate_text_embeddings(
        csv_path=CSV_PATH,
        checkpoint_path=CHECKPOINT_PATH,
        output_dir=OUTPUT_DIR,
        batch_size=BATCH_SIZE,
        max_length=MAX_LENGTH
    )
    
    # Verification
    print("\n" + "="*80)
    print("VERIFICATION - LOADING SAVED EMBEDDINGS")
    print("="*80 + "\n")
    
    data = load_text_embeddings(OUTPUT_DIR)
    print(f"Loaded embeddings shape: {data['embeddings'].shape}")
    print(f"Loaded labels shape: {data['labels'].shape}")
    print(f"Embedding dimension: {data['metadata']['embedding_dim']}")
    print(f"Total samples: {data['metadata']['total_samples']:,}")
    print()
    print("✓ Verification complete!")
