"""
YOLO Model Training Script
Trains YOLOv8 model on the prepared dataset
"""

import os
from pathlib import Path
from ultralytics import YOLO
import torch

# Configuration
BASE_DIR = Path(__file__).parent
DATASET_CONFIG = BASE_DIR / "yolo_dataset" / "data.yaml"
OUTPUT_DIR = BASE_DIR / "runs"

# Training parameters
MODEL_SIZE = 'yolov8n.pt'  # Options: yolov8n.pt, yolov8s.pt, yolov8m.pt, yolov8l.pt, yolov8x.pt
EPOCHS = 100
BATCH_SIZE = 16
IMAGE_SIZE = 640
DEVICE = 0 if torch.cuda.is_available() else 'cpu'

def train_model():
    """Train YOLO model on the prepared dataset."""
    
    print("=" * 60)
    print("YOLO Model Training")
    print("=" * 60)
    
    # Check if dataset config exists
    if not DATASET_CONFIG.exists():
        print(f"\n❌ ERROR: Dataset configuration not found!")
        print(f"Expected location: {DATASET_CONFIG}")
        print("\nPlease run 'prepare_dataset.py' first to prepare the dataset.")
        return
    
    # Print configuration
    print(f"\n[Configuration]")
    print(f"  Model: {MODEL_SIZE}")
    print(f"  Epochs: {EPOCHS}")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Image size: {IMAGE_SIZE}")
    print(f"  Device: {DEVICE} ({'GPU' if DEVICE != 'cpu' else 'CPU'})")
    print(f"  Dataset config: {DATASET_CONFIG}")
    
    # Load a pre-trained model
    print(f"\n[1/3] Loading {MODEL_SIZE} model...")
    model = YOLO(MODEL_SIZE)
    
    # Train the model
    print(f"\n[2/3] Starting training for {EPOCHS} epochs...")
    print("-" * 60)
    
    results = model.train(
        data=str(DATASET_CONFIG),
        epochs=EPOCHS,
        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,
        device=DEVICE,
        project=str(OUTPUT_DIR),
        name='train',
        patience=50,  # Early stopping patience
        save=True,
        save_period=10,  # Save checkpoint every 10 epochs
        plots=True,  # Save plots
        verbose=True,
        exist_ok=True
    )
    
    print("-" * 60)
    print("\n[3/3] Training complete!")
    
    # Get best model path
    best_model_path = OUTPUT_DIR / "train" / "weights" / "best.pt"
    last_model_path = OUTPUT_DIR / "train" / "weights" / "last.pt"
    
    print("\n" + "=" * 60)
    print("✅ Training Complete!")
    print("=" * 60)
    print(f"\nBest model saved at: {best_model_path}")
    print(f"Last model saved at: {last_model_path}")
    print(f"Training results: {OUTPUT_DIR / 'train'}")
    print("\nNext steps:")
    print("  1. Run 'evaluate_model.py' to evaluate the model")
    print("  2. Run 'test_custom_images.py' to test on custom images")
    
    return model, results

if __name__ == "__main__":
    # Check for GPU
    if torch.cuda.is_available():
        print(f"\n✓ GPU available: {torch.cuda.get_device_name(0)}")
    else:
        print("\n⚠ No GPU detected. Training will use CPU (slower).")
    
    train_model()
