"""
Installation Verification Script
Checks if all required packages are installed correctly
"""

import sys
from pathlib import Path

def check_package(package_name, import_name=None):
    """Check if a package is installed."""
    if import_name is None:
        import_name = package_name
    
    try:
        __import__(import_name)
        return True, "✓ Installed"
    except ImportError:
        return False, "✗ Missing"

def verify_installation():
    """Verify all required packages are installed."""
    
    print("=" * 70)
    print(" " * 20 + "INSTALLATION VERIFICATION")
    print("=" * 70)
    
    print(f"\nPython Version: {sys.version.split()[0]}")
    
    # Required packages
    packages = [
        ("ultralytics", "ultralytics"),
        ("torch", "torch"),
        ("torchvision", "torchvision"),
        ("opencv-python", "cv2"),
        ("matplotlib", "matplotlib"),
        ("seaborn", "seaborn"),
        ("pandas", "pandas"),
        ("numpy", "numpy"),
        ("PIL", "PIL"),
        ("yaml", "yaml"),
        ("tqdm", "tqdm"),
    ]
    
    print("\n" + "-" * 70)
    print("Checking Required Packages:")
    print("-" * 70)
    
    all_installed = True
    missing_packages = []
    
    for package_name, import_name in packages:
        installed, status = check_package(package_name, import_name)
        print(f"  {status} {package_name}")
        
        if not installed:
            all_installed = False
            missing_packages.append(package_name)
    
    # Check CUDA availability
    print("\n" + "-" * 70)
    print("GPU/CUDA Check:")
    print("-" * 70)
    
    try:
        import torch
        if torch.cuda.is_available():
            print(f"  ✓ CUDA available")
            print(f"    GPU: {torch.cuda.get_device_name(0)}")
            print(f"    CUDA version: {torch.version.cuda}")
        else:
            print(f"  ○ CUDA not available (will use CPU)")
            print(f"    Note: Training will be slower without GPU")
    except:
        print(f"  ✗ Cannot check CUDA (torch not installed)")
    
    # Check file structure
    print("\n" + "-" * 70)
    print("File Structure Check:")
    print("-" * 70)
    
    base_dir = Path(__file__).parent
    
    required_files = [
        "prepare_dataset.py",
        "train_yolo.py",
        "evaluate_model.py",
        "test_custom_images.py",
        "run_pipeline.py",
        "requirements.txt"
    ]
    
    required_dirs = [
        "Fooled You Labelled Images",
        "Numex Labelled Images"
    ]
    
    for file_name in required_files:
        file_path = base_dir / file_name
        if file_path.exists():
            print(f"  ✓ {file_name}")
        else:
            print(f"  ✗ {file_name} (missing)")
    
    for dir_name in required_dirs:
        dir_path = base_dir / dir_name
        if dir_path.exists():
            txt_files = list(dir_path.glob("*.txt"))
            print(f"  ✓ {dir_name}/ ({len(txt_files)} label files)")
        else:
            print(f"  ✗ {dir_name}/ (missing)")
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    if all_installed:
        print("\n✅ All required packages are installed!")
    else:
        print(f"\n⚠ {len(missing_packages)} package(s) missing!")
        print("\nMissing packages:")
        for pkg in missing_packages:
            print(f"  - {pkg}")
        print("\nTo install missing packages, run:")
        print("  pip install -r requirements.txt")
    
    print("\n" + "=" * 70)
    print("NEXT STEPS")
    print("=" * 70)
    
    if all_installed:
        print("\n1. Check if images are available:")
        print("   python check_images.py")
        print("\n2. Run the complete pipeline:")
        print("   python run_pipeline.py")
        print("\nOr read QUICK_START.md for more options.")
    else:
        print("\n1. Install missing packages:")
        print("   pip install -r requirements.txt")
        print("\n2. Run this script again to verify:")
        print("   python verify_installation.py")
    
    print("\n" + "=" * 70)
    
    return all_installed

if __name__ == "__main__":
    try:
        success = verify_installation()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ Error during verification: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
