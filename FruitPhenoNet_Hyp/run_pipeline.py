"""
Complete Pipeline Runner
Runs all steps: dataset preparation, training, evaluation, and testing
"""

import sys
from pathlib import Path

def run_complete_pipeline():
    """Run the complete YOLO training pipeline."""
    
    print("=" * 80)
    print(" " * 20 + "FRUITPHENONET - YOLO TRAINING PIPELINE")
    print("=" * 80)
    print("\nThis script will run the complete pipeline:")
    print("  1. Dataset Preparation")
    print("  2. Model Training")
    print("  3. Model Evaluation")
    print("  4. Custom Image Testing")
    print("\n" + "=" * 80)
    
    response = input("\nDo you want to continue? (yes/no) [yes]: ").strip().lower()
    if response and response not in ['yes', 'y']:
        print("Pipeline cancelled.")
        return
    
    # Step 1: Dataset Preparation
    print("\n" + "=" * 80)
    print("STEP 1/4: DATASET PREPARATION")
    print("=" * 80)
    
    try:
        from prepare_dataset import prepare_dataset
        prepare_dataset()
    except Exception as e:
        print(f"\n❌ Error in dataset preparation: {e}")
        print("Pipeline stopped.")
        return
    
    print("\n✓ Dataset preparation complete!")
    input("\nPress Enter to continue to training...")
    
    # Step 2: Model Training
    print("\n" + "=" * 80)
    print("STEP 2/4: MODEL TRAINING")
    print("=" * 80)
    print("\n⚠ Note: Training may take a while depending on your hardware.")
    print("  - With GPU: ~15-30 minutes for 100 epochs")
    print("  - With CPU: ~2-4 hours for 100 epochs")
    
    response = input("\nContinue with training? (yes/no) [yes]: ").strip().lower()
    if response and response not in ['yes', 'y']:
        print("Training skipped. You can run 'python train_yolo.py' later.")
        return
    
    try:
        from train_yolo import train_model
        train_model()
    except Exception as e:
        print(f"\n❌ Error in training: {e}")
        print("Pipeline stopped.")
        return
    
    print("\n✓ Model training complete!")
    input("\nPress Enter to continue to evaluation...")
    
    # Step 3: Model Evaluation
    print("\n" + "=" * 80)
    print("STEP 3/4: MODEL EVALUATION")
    print("=" * 80)
    
    try:
        from evaluate_model import evaluate_model
        evaluate_model()
    except Exception as e:
        print(f"\n❌ Error in evaluation: {e}")
        print("Pipeline stopped.")
        return
    
    print("\n✓ Model evaluation complete!")
    input("\nPress Enter to continue to custom image testing...")
    
    # Step 4: Custom Image Testing
    print("\n" + "=" * 80)
    print("STEP 4/4: CUSTOM IMAGE TESTING")
    print("=" * 80)
    
    print("\nChoose testing option:")
    print("  1. Test on custom images (from 'custom_test_images' folder)")
    print("  2. Test on validation set")
    print("  3. Skip testing (you can run 'python test_custom_images.py' later)")
    
    choice = input("\nEnter choice (1/2/3) [3]: ").strip() or "3"
    
    if choice == "3":
        print("\nTesting skipped.")
    else:
        try:
            from test_custom_images import test_custom_images, test_on_validation_set
            
            if choice == "1":
                test_custom_images()
            elif choice == "2":
                test_on_validation_set()
        except Exception as e:
            print(f"\n❌ Error in testing: {e}")
    
    # Final Summary
    print("\n" + "=" * 80)
    print("✅ PIPELINE COMPLETE!")
    print("=" * 80)
    print("\nGenerated files:")
    print("  📁 yolo_dataset/          - Prepared dataset")
    print("  📁 runs/train/            - Training outputs & trained models")
    print("  📁 evaluation_reports/    - Accuracy reports")
    print("  📁 test_results/          - Testing outputs")
    print("\nNext steps:")
    print("  • Review accuracy report in 'evaluation_reports/'")
    print("  • Check training plots in 'runs/train/'")
    print("  • Test on your own images: add to 'custom_test_images/' and run test_custom_images.py")
    print("\n" + "=" * 80)

if __name__ == "__main__":
    try:
        run_complete_pipeline()
    except KeyboardInterrupt:
        print("\n\nPipeline interrupted by user.")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
