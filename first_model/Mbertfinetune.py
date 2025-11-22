import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    TrainingArguments, 
    Trainer,
    EarlyStoppingCallback
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
import os
from datetime import datetime

# Set random seeds for reproducibility
def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

# Custom Dataset class
class SuicideDetectionDataset(Dataset):
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
            'labels': torch.tensor(label, dtype=torch.long)
        }

# Compute metrics for evaluation
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

def main():
    print("="*80)
    print("MENTALBERT FINE-TUNING FOR SUICIDE DETECTION")
    print("="*80)
    print()
    
    # Configuration
    # Mental health-focused BERT models (in order of preference):
    # 1. "mental/mental-bert-base-uncased" - MentalBERT (GATED - requires access request)
    # 2. "mental/mental-roberta-base" - Mental RoBERTa (alternative)
    # 3. "bert-base-uncased" - Standard BERT (fallback)
    
    # Try mental health models, fallback to BERT if unavailable
    MODELS_TO_TRY = [
        "mental/mentalbert-base-uncased",
        "mental/mental-roberta-base",  # Mental RoBERTa (may be available)
        "emilyalsentzer/Bio_ClinicalBERT",  # Clinical BERT for medical text
        "bert-base-uncased"  # Standard BERT fallback
    ]
    
    MAX_LENGTH = 512
    # Optimized batch sizes for RTX 3050 (6GB VRAM)
    BATCH_SIZE = 12  # Optimized for 6GB VRAM - good balance of speed and memory
    GRADIENT_ACCUMULATION_STEPS = 2  # Effective batch size = 12 * 2 = 24
    LEARNING_RATE = 2e-5
    NUM_EPOCHS = 3
    WARMUP_STEPS = 500
    WEIGHT_DECAY = 0.01
    
    # Check for GPU and optimize settings
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"Total GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        
        # Enable GPU optimizations
        torch.backends.cudnn.benchmark = True
        torch.cuda.empty_cache()
        print("✓ GPU optimizations enabled")
    else:
        print("⚠ No GPU detected - training will be slow on CPU")
        GRADIENT_ACCUMULATION_STEPS = 1  # No gradient accumulation needed on CPU
    print()
    
    # Load data
    print("Loading cleaned dataset...")
    df = pd.read_csv('Suicide_Detection_cleaned.csv', usecols=['text', 'class'])
    
    # Clean up the class column (remove extra whitespace/newlines)
    df['class'] = df['class'].str.strip()
    
    # Filter out any corrupted rows (keep only 'suicide' and 'non-suicide')
    df = df[df['class'].isin(['suicide', 'non-suicide'])]
    
    print(f"Total samples: {len(df)}")
    print(f"\nClass distribution:")
    print(df['class'].value_counts())
    print()
    
    # Convert labels to binary (0: non-suicide, 1: suicide)
    label_map = {'non-suicide': 0, 'suicide': 1}
    df['label'] = df['class'].map(label_map)
    
    # Split data: 80% train, 10% validation, 10% test
    train_texts, temp_texts, train_labels, temp_labels = train_test_split(
        df['text'].values,
        df['label'].values,
        test_size=0.2,
        random_state=42,
        stratify=df['label'].values
    )
    
    val_texts, test_texts, val_labels, test_labels = train_test_split(
        temp_texts,
        temp_labels,
        test_size=0.5,
        random_state=42,
        stratify=temp_labels
    )
    
    print(f"Train set: {len(train_texts)} samples")
    print(f"Validation set: {len(val_texts)} samples")
    print(f"Test set: {len(test_texts)} samples")
    print()
    
    # Load mental health-focused model (try in order of preference)
    print("Loading mental health-focused BERT model from HuggingFace Hub...")
    print("Trying models in order of preference:")
    for model_name in MODELS_TO_TRY:
        print(f"  - {model_name}")
    print()
    
    MODEL_NAME = None
    tokenizer = None
    model = None
    
    for model_candidate in MODELS_TO_TRY:
        try:
            print(f"Attempting to load: {model_candidate}...")
            tokenizer = AutoTokenizer.from_pretrained(model_candidate)
            model = AutoModelForSequenceClassification.from_pretrained(
                model_candidate,
                num_labels=2,
                problem_type="single_label_classification",
                ignore_mismatched_sizes=True  # In case the classifier head needs to be replaced
            )
            MODEL_NAME = model_candidate
            print(f"✓ Successfully loaded: {MODEL_NAME}")
            print(f"✓ Number of parameters: {sum(p.numel() for p in model.parameters()):,}")
            print()
            break
        except Exception as e:
            print(f"✗ Could not load {model_candidate}: {str(e)[:100]}")
            print()
            continue
    
    if MODEL_NAME is None:
        raise RuntimeError("Could not load any model. Please check your internet connection and try again.")
    
    # Move model to GPU if available
    if torch.cuda.is_available():
        model = model.to(device)
        print(f"✓ Model moved to GPU: {device}")
        print(f"✓ GPU Memory allocated: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
        print()
    
    # Create datasets
    print("Creating datasets...")
    train_dataset = SuicideDetectionDataset(train_texts, train_labels, tokenizer, MAX_LENGTH)
    val_dataset = SuicideDetectionDataset(val_texts, val_labels, tokenizer, MAX_LENGTH)
    test_dataset = SuicideDetectionDataset(test_texts, test_labels, tokenizer, MAX_LENGTH)
    print("Datasets created!")
    print()
    
    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"./mentalbert_suicide_detection_{timestamp}"
    
    # Training arguments - Optimized for RTX 3050
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,  # Effective larger batch size
        warmup_steps=WARMUP_STEPS,
        weight_decay=WEIGHT_DECAY,
        learning_rate=LEARNING_RATE,
        logging_dir=f'{output_dir}/logs',
        logging_steps=100,
        eval_strategy="steps",
        eval_steps=500,
        save_strategy="steps",
        save_steps=500,
        save_total_limit=3,  # Keep only 3 checkpoints to save disk space
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        report_to="none",
        # GPU optimizations for RTX 3050
        fp16=torch.cuda.is_available(),  # Mixed precision - reduces memory by ~50%
        fp16_opt_level="O1",  # Moderate mixed precision
        dataloader_num_workers=0,  # Set to 0 to avoid multiprocessing issues on Windows
        dataloader_pin_memory=torch.cuda.is_available(),  # Pin memory for faster GPU transfer
        gradient_checkpointing=False,  # Disabled for speed (enable if OOM occurs)
        optim="adamw_torch",  # Use PyTorch's AdamW (more memory efficient)
        # Memory management
        max_grad_norm=1.0,  # Gradient clipping
        eval_accumulation_steps=1,  # Reduce memory during evaluation
    )
    
    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )
    
    # Train the model
    print("="*80)
    print("STARTING TRAINING")
    print("="*80)
    if torch.cuda.is_available():
        print(f"GPU Memory before training: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
        print(f"Effective batch size: {BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS}")
    print()
    
    train_result = trainer.train()
    
    # Clear GPU cache after training
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"\n✓ GPU Memory after training: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
    
    print()
    print("="*80)
    print("TRAINING COMPLETED")
    print("="*80)
    print()
    
    # Save the final model
    final_model_dir = f"{output_dir}/final_model"
    trainer.save_model(final_model_dir)
    tokenizer.save_pretrained(final_model_dir)
    print(f"✓ Final model saved to: {final_model_dir}")
    print()
    
    # Evaluate on validation set
    print("="*80)
    print("VALIDATION SET EVALUATION")
    print("="*80)
    val_results = trainer.evaluate(eval_dataset=val_dataset)
    print("\nValidation Metrics:")
    for key, value in val_results.items():
        print(f"  {key}: {value:.4f}")
    print()
    
    # Evaluate on test set
    print("="*80)
    print("TEST SET EVALUATION")
    print("="*80)
    
    predictions = trainer.predict(test_dataset)
    pred_labels = predictions.predictions.argmax(-1)
    
    # Calculate detailed metrics
    accuracy = accuracy_score(test_labels, pred_labels)
    precision, recall, f1, _ = precision_recall_fscore_support(
        test_labels, pred_labels, average='binary'
    )
    
    print("\nTest Set Metrics:")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1 Score:  {f1:.4f}")
    print()
    
    # Confusion matrix
    cm = confusion_matrix(test_labels, pred_labels)
    print("Confusion Matrix:")
    print("                  Predicted")
    print("                Non-Suicide  Suicide")
    print(f"Actual Non-Sui      {cm[0][0]:5d}      {cm[0][1]:5d}")
    print(f"       Suicide      {cm[1][0]:5d}      {cm[1][1]:5d}")
    print()
    
    # Classification report
    print("Detailed Classification Report:")
    print(classification_report(
        test_labels, 
        pred_labels, 
        target_names=['non-suicide', 'suicide'],
        digits=4
    ))
    
    # Save results to file
    results_file = f"{output_dir}/evaluation_results.txt"
    with open(results_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("MENTALBERT SUICIDE DETECTION - EVALUATION RESULTS\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"Model: {MODEL_NAME}\n")
        f.write(f"Training samples: {len(train_texts)}\n")
        f.write(f"Validation samples: {len(val_texts)}\n")
        f.write(f"Test samples: {len(test_texts)}\n\n")
        
        f.write("Validation Metrics:\n")
        for key, value in val_results.items():
            f.write(f"  {key}: {value:.4f}\n")
        f.write("\n")
        
        f.write("Test Set Metrics:\n")
        f.write(f"  Accuracy:  {accuracy:.4f}\n")
        f.write(f"  Precision: {precision:.4f}\n")
        f.write(f"  Recall:    {recall:.4f}\n")
        f.write(f"  F1 Score:  {f1:.4f}\n\n")
        
        f.write("Confusion Matrix:\n")
        f.write("                  Predicted\n")
        f.write("                Non-Suicide  Suicide\n")
        f.write(f"Actual Non-Sui      {cm[0][0]:5d}      {cm[0][1]:5d}\n")
        f.write(f"       Suicide      {cm[1][0]:5d}      {cm[1][1]:5d}\n\n")
        
        f.write("Classification Report:\n")
        f.write(classification_report(
            test_labels, 
            pred_labels, 
            target_names=['non-suicide', 'suicide'],
            digits=4
        ))
    
    print(f"✓ Results saved to: {results_file}")
    print()
    
    # Save sample predictions
    print("Sample Predictions:")
    print("-"*80)
    sample_indices = np.random.choice(len(test_texts), min(10, len(test_texts)), replace=False)
    for idx in sample_indices:
        text = test_texts[idx][:100] + "..." if len(test_texts[idx]) > 100 else test_texts[idx]
        true_label = "suicide" if test_labels[idx] == 1 else "non-suicide"
        pred_label = "suicide" if pred_labels[idx] == 1 else "non-suicide"
        status = "✓" if test_labels[idx] == pred_labels[idx] else "✗"
        print(f"{status} Text: {text}")
        print(f"  True: {true_label} | Predicted: {pred_label}")
        print()
    
    print("="*80)
    print("TRAINING AND EVALUATION COMPLETE!")
    print("="*80)
    print()
    print(f"Model checkpoints saved in: {output_dir}")
    print(f"Final model saved in: {final_model_dir}")
    print(f"Evaluation results saved in: {results_file}")
    
    # Final GPU stats
    if torch.cuda.is_available():
        print(f"\nFinal GPU Memory Usage: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
        print(f"Peak GPU Memory Usage: {torch.cuda.max_memory_allocated(0) / 1024**3:.2f} GB")
        torch.cuda.empty_cache()
    print()

if __name__ == "__main__":
    main()
