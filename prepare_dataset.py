"""
Dataset Preparation Script for YOLO Training
Organizes images and labels into train/val/test splits
"""

import os
import shutil
import random
from pathlib import Path
from collections import defaultdict

# Set random seed for reproducibility
random.seed(42)

# Configuration
BASE_DIR = Path(__file__).parent
FOOLED_YOU_DIR = BASE_DIR / "Fooled You Labelled Images"
NUMEX_DIR = BASE_DIR / "Numex Labelled Images"
OUTPUT_DIR = BASE_DIR / "yolo_dataset"

# Split ratios
TRAIN_RATIO = 0.7
VAL_RATIO = 0.2
TEST_RATIO = 0.1

def find_image_for_label(label_file, search_dirs):
    """Find the corresponding image file for a label file."""
    label_stem = label_file.stem  # filename without extension
    
    # Common image extensions
    image_extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']
    
    for search_dir in search_dirs:
        for ext in image_extensions:
            image_path = search_dir / f"{label_stem}{ext}"
            if image_path.exists():
                return image_path
    return None

def prepare_dataset():
    """Prepare YOLO dataset structure with train/val/test splits."""
    
    print("=" * 60)
    print("YOLO Dataset Preparation")
    print("=" * 60)
    
    # Create output directory structure
    for split in ['train', 'val', 'test']:
        (OUTPUT_DIR / split / 'images').mkdir(parents=True, exist_ok=True)
        (OUTPUT_DIR / split / 'labels').mkdir(parents=True, exist_ok=True)
    
    # Collect all label files
    all_data = []
    
    print("\n[1/5] Collecting Fooled You data...")
    fooled_you_labels = list(FOOLED_YOU_DIR.glob("*.txt"))
    fooled_you_labels = [f for f in fooled_you_labels if f.name != "classes.txt"]
    
    # Try to find images in parent directory or same directory
    search_dirs = [FOOLED_YOU_DIR, FOOLED_YOU_DIR.parent / "Fooled You Images"]
    
    for label_file in fooled_you_labels:
        image_file = find_image_for_label(label_file, search_dirs)
        if image_file:
            all_data.append({
                'label': label_file,
                'image': image_file,
                'class_name': 'fooled_you'
            })
        else:
            print(f"  Warning: No image found for {label_file.name}")
    
    print(f"  Found {len([d for d in all_data if d['class_name'] == 'fooled_you'])} Fooled You samples")
    
    print("\n[2/5] Collecting Numex data...")
    numex_labels = list(NUMEX_DIR.glob("*.txt"))
    numex_labels = [f for f in numex_labels if f.name != "classes.txt"]
    
    search_dirs = [NUMEX_DIR, NUMEX_DIR.parent / "Numex Images"]
    
    for label_file in numex_labels:
        image_file = find_image_for_label(label_file, search_dirs)
        if image_file:
            all_data.append({
                'label': label_file,
                'image': image_file,
                'class_name': 'numex'
            })
        else:
            print(f"  Warning: No image found for {label_file.name}")
    
    print(f"  Found {len([d for d in all_data if d['class_name'] == 'numex'])} Numex samples")
    
    if not all_data:
        print("\n❌ ERROR: No valid image-label pairs found!")
        print("\nPlease ensure:")
        print("  1. Image files (.jpg, .png) exist alongside label files (.txt)")
        print("  2. Image and label files have matching names")
        print(f"  3. Images are in: {FOOLED_YOU_DIR} or {FOOLED_YOU_DIR.parent / 'Fooled You Images'}")
        print(f"                    {NUMEX_DIR} or {NUMEX_DIR.parent / 'Numex Images'}")
        return
    
    print(f"\n[3/5] Total samples found: {len(all_data)}")
    
    # Shuffle data
    random.shuffle(all_data)
    
    # Split data
    total = len(all_data)
    train_end = int(total * TRAIN_RATIO)
    val_end = train_end + int(total * VAL_RATIO)
    
    train_data = all_data[:train_end]
    val_data = all_data[train_end:val_end]
    test_data = all_data[val_end:]
    
    print(f"\n[4/5] Data split:")
    print(f"  Train: {len(train_data)} samples ({len(train_data)/total*100:.1f}%)")
    print(f"  Val:   {len(val_data)} samples ({len(val_data)/total*100:.1f}%)")
    print(f"  Test:  {len(test_data)} samples ({len(test_data)/total*100:.1f}%)")
    
    # Copy files to respective directories
    print("\n[5/5] Copying files...")
    
    splits = {
        'train': train_data,
        'val': val_data,
        'test': test_data
    }
    
    for split_name, data_list in splits.items():
        print(f"  Copying {split_name} data...")
        for idx, item in enumerate(data_list, 1):
            # Copy image
            img_dest = OUTPUT_DIR / split_name / 'images' / item['image'].name
            shutil.copy2(item['image'], img_dest)
            
            # Copy label
            label_dest = OUTPUT_DIR / split_name / 'labels' / item['label'].name
            shutil.copy2(item['label'], label_dest)
    
    # Create classes file
    print("\n[6/6] Creating dataset configuration...")
    
    # Read class names from both directories
    classes = []
    if (FOOLED_YOU_DIR / "classes.txt").exists():
        with open(FOOLED_YOU_DIR / "classes.txt", 'r') as f:
            fooled_class = f.read().strip()
            if fooled_class:
                classes.append(fooled_class)
    
    if (NUMEX_DIR / "classes.txt").exists():
        with open(NUMEX_DIR / "classes.txt", 'r') as f:
            numex_class = f.read().strip()
            if numex_class:
                classes.append(numex_class)
    
    # If no classes found in files, use defaults
    if not classes:
        classes = ['bell-pepper(fooled_you)', 'bell-pepper(numex)']
    
    # Create data.yaml for YOLO
    yaml_content = f"""# FruitPhenoNet Dataset Configuration
# Generated for YOLO training

path: {OUTPUT_DIR.absolute().as_posix()}  # dataset root dir
train: train/images  # train images (relative to 'path')
val: val/images  # val images (relative to 'path')
test: test/images  # test images (relative to 'path')

# Classes
nc: {len(classes)}  # number of classes
names: {classes}  # class names
"""
    
    with open(OUTPUT_DIR / "data.yaml", 'w') as f:
        f.write(yaml_content)
    
    print(f"  ✓ Created data.yaml with {len(classes)} classes")
    print(f"  ✓ Dataset ready at: {OUTPUT_DIR}")
    
    print("\n" + "=" * 60)
    print("✅ Dataset preparation complete!")
    print("=" * 60)
    print(f"\nDataset location: {OUTPUT_DIR}")
    print(f"Configuration file: {OUTPUT_DIR / 'data.yaml'}")
    print("\nNext step: Run train_yolo.py to train the model")

if __name__ == "__main__":
    prepare_dataset()
