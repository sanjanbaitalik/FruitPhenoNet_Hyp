"""
Check if images exist for the label files
This helps diagnose if the dataset is ready for training
"""

import os
from pathlib import Path
from collections import defaultdict

BASE_DIR = Path(__file__).parent
FOOLED_YOU_DIR = BASE_DIR / "Fooled You Labelled Images"
NUMEX_DIR = BASE_DIR / "Numex Labelled Images"

def check_images():
    """Check if image files exist for label files."""
    
    print("=" * 80)
    print("IMAGE AVAILABILITY CHECK")
    print("=" * 80)
    
    # Image extensions to look for
    image_extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']
    
    # Search directories
    search_dirs_fooled = [
        FOOLED_YOU_DIR,
        FOOLED_YOU_DIR.parent / "Fooled You Images",
        BASE_DIR / "Fooled You Images"
    ]
    
    search_dirs_numex = [
        NUMEX_DIR,
        NUMEX_DIR.parent / "Numex Images",
        BASE_DIR / "Numex Images"
    ]
    
    # Check Fooled You
    print("\n[1/2] Checking Fooled You Labelled Images...")
    print(f"Label directory: {FOOLED_YOU_DIR}")
    
    fooled_labels = list(FOOLED_YOU_DIR.glob("*.txt"))
    fooled_labels = [f for f in fooled_labels if f.name != "classes.txt"]
    
    print(f"Found {len(fooled_labels)} label files")
    
    fooled_found = 0
    fooled_missing = []
    
    for label_file in fooled_labels:
        label_stem = label_file.stem
        found = False
        
        for search_dir in search_dirs_fooled:
            if not search_dir.exists():
                continue
            for ext in image_extensions:
                image_path = search_dir / f"{label_stem}{ext}"
                if image_path.exists():
                    fooled_found += 1
                    found = True
                    break
            if found:
                break
        
        if not found:
            fooled_missing.append(label_stem)
    
    print(f"\n✓ Images found: {fooled_found}/{len(fooled_labels)}")
    if fooled_missing:
        print(f"✗ Missing images: {len(fooled_missing)}")
        if len(fooled_missing) <= 10:
            print("  Missing for labels:")
            for name in fooled_missing[:10]:
                print(f"    - {name}")
    
    # Check Numex
    print("\n[2/2] Checking Numex Labelled Images...")
    print(f"Label directory: {NUMEX_DIR}")
    
    numex_labels = list(NUMEX_DIR.glob("*.txt"))
    numex_labels = [f for f in numex_labels if f.name != "classes.txt"]
    
    print(f"Found {len(numex_labels)} label files")
    
    numex_found = 0
    numex_missing = []
    
    for label_file in numex_labels:
        label_stem = label_file.stem
        found = False
        
        for search_dir in search_dirs_numex:
            if not search_dir.exists():
                continue
            for ext in image_extensions:
                image_path = search_dir / f"{label_stem}{ext}"
                if image_path.exists():
                    numex_found += 1
                    found = True
                    break
            if found:
                break
        
        if not found:
            numex_missing.append(label_stem)
    
    print(f"\n✓ Images found: {numex_found}/{len(numex_labels)}")
    if numex_missing:
        print(f"✗ Missing images: {len(numex_missing)}")
        if len(numex_missing) <= 10:
            print("  Missing for labels:")
            for name in numex_missing[:10]:
                print(f"    - {name}")
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    total_labels = len(fooled_labels) + len(numex_labels)
    total_found = fooled_found + numex_found
    total_missing = len(fooled_missing) + len(numex_missing)
    
    print(f"\nTotal label files: {total_labels}")
    print(f"Total images found: {total_found}")
    print(f"Total images missing: {total_missing}")
    
    if total_missing == 0:
        print("\n✅ All images found! You're ready to proceed with training.")
    else:
        print(f"\n⚠ {total_missing} images are missing.")
        print("\nImages should be in one of these locations:")
        print("\nFor Fooled You:")
        for dir in search_dirs_fooled:
            print(f"  - {dir}")
        print("\nFor Numex:")
        for dir in search_dirs_numex:
            print(f"  - {dir}")
        print("\nImage files should have the same name as label files but with")
        print("image extensions (.jpg, .jpeg, .png)")
        print("\nExample:")
        print("  Label: 105_0_0_14-9-22.txt")
        print("  Image: 105_0_0_14-9-22.jpg")
    
    print("\n" + "=" * 80)
    
    return {
        'total_labels': total_labels,
        'total_found': total_found,
        'total_missing': total_missing,
        'ready': total_missing == 0
    }

if __name__ == "__main__":
    result = check_images()
    
    if result['ready']:
        print("\n✅ Ready to train! Run: python run_pipeline.py")
    else:
        print("\n⚠ Please add missing images before training.")
