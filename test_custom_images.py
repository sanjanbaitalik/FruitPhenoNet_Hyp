"""
Custom Image Testing Script
Test the trained YOLO model on custom images and visualize results
"""

import os
from pathlib import Path
from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np

# Configuration
BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "runs" / "train" / "weights" / "best.pt"
TEST_IMAGES_DIR = BASE_DIR / "custom_test_images"
OUTPUT_DIR = BASE_DIR / "test_results"

def test_custom_images(image_paths=None, confidence_threshold=0.25):
    """
    Test model on custom images.
    
    Args:
        image_paths: List of image paths or directory path. If None, uses TEST_IMAGES_DIR
        confidence_threshold: Confidence threshold for detections (0-1)
    """
    
    print("=" * 60)
    print("Custom Image Testing")
    print("=" * 60)
    
    # Check if model exists
    if not MODEL_PATH.exists():
        print(f"\n❌ ERROR: Trained model not found!")
        print(f"Expected location: {MODEL_PATH}")
        print("\nPlease run 'train_yolo.py' first to train the model.")
        return
    
    # Load model
    print(f"\n[1/4] Loading model from: {MODEL_PATH}")
    model = YOLO(str(MODEL_PATH))
    
    # Get test images
    if image_paths is None:
        # Create test images directory if it doesn't exist
        TEST_IMAGES_DIR.mkdir(exist_ok=True)
        
        # Check if directory has images
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
        test_images = []
        for ext in image_extensions:
            test_images.extend(list(TEST_IMAGES_DIR.glob(ext)))
        
        if not test_images:
            print(f"\n⚠ No test images found in: {TEST_IMAGES_DIR}")
            print("\nPlease add some test images to the 'custom_test_images' folder")
            print("and run this script again.")
            print("\nAlternatively, you can call test_custom_images() with specific image paths:")
            print("  test_custom_images(image_paths=['path/to/image1.jpg', 'path/to/image2.jpg'])")
            return
    else:
        # Handle single image path or list of paths
        if isinstance(image_paths, (str, Path)):
            image_paths = [image_paths]
        
        test_images = [Path(p) for p in image_paths if Path(p).exists()]
        
        if not test_images:
            print(f"\n❌ ERROR: No valid image paths provided!")
            return
    
    print(f"\n[2/4] Found {len(test_images)} test images")
    
    # Create output directory
    OUTPUT_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir = OUTPUT_DIR / f"test_session_{timestamp}"
    session_dir.mkdir(exist_ok=True)
    
    print(f"\n[3/4] Running predictions (confidence threshold: {confidence_threshold})...")
    
    # Process each image
    results_summary = []
    
    for idx, image_path in enumerate(test_images, 1):
        print(f"  Processing {idx}/{len(test_images)}: {image_path.name}")
        
        # Run inference
        results = model.predict(
            source=str(image_path),
            conf=confidence_threshold,
            save=False,
            verbose=False
        )
        
        # Get results
        result = results[0]
        detections = len(result.boxes)
        
        # Load original image (we'll draw our own compact labels)
        img = cv2.imread(str(image_path))
        if img is None:
            print(f"    ❌ Could not read image: {image_path}")
            continue

        # If there are detections, draw boxes and short single-key labels
        if detections > 0:
            # Extract boxes, classes and confidences
            try:
                boxes = result.boxes.xyxy.cpu().numpy()
            except Exception:
                # Fallback if attribute naming differs
                boxes = result.boxes.xyxy.numpy()

            classes = result.boxes.cls.cpu().numpy().astype(int)
            confidences = result.boxes.conf.cpu().numpy()

            # Drawing parameters
            img_h, img_w = img.shape[:2]
            thickness = max(1, int(round(min(img_w, img_h) / 500.0)))

            for (x1, y1, x2, y2), cls, conf in zip(boxes, classes, confidences):
                x1i, y1i, x2i, y2i = int(x1), int(y1), int(x2), int(y2)

                # box color - using a fixed color (B, G, R format for OpenCV)
                # Options: (0, 255, 0) = Green, (255, 0, 0) = Blue, (0, 0, 255) = Red
                # (0, 255, 255) = Yellow, (255, 0, 255) = Magenta, (255, 255, 0) = Cyan
                color = (0, 255, 255)  # Green boxes

                # draw rectangle only (no labels)
                cv2.rectangle(img, (x1i, y1i), (x2i, y2i), color, thickness)

            annotated_img = img
        else:
            # No detections: save the original image unchanged
            annotated_img = img

        output_path = session_dir / f"result_{image_path.stem}.jpg"
        cv2.imwrite(str(output_path), annotated_img)
        
        # Collect detection details
        detection_info = {
            'image': image_path.name,
            'detections': detections,
            'output_path': output_path
        }
        
        if detections > 0:
            # classes and confidences were already extracted above when drawing
            # Extract class names for the summary printout
            try:
                classes = result.boxes.cls.cpu().numpy().astype(int)
                confidences = result.boxes.conf.cpu().numpy()
            except Exception:
                classes = []
                confidences = []

            class_names = [model.names[int(c)] for c in classes]
            detection_info['classes'] = class_names
            detection_info['confidences'] = confidences

            print(f"    ✓ Detected {detections} object(s)")
            for cls_name, conf in zip(class_names, confidences):
                print(f"      - {cls_name}: {conf:.2%} confidence")
        else:
            print(f"    ○ No objects detected")
            detection_info['classes'] = []
            detection_info['confidences'] = []
        
        results_summary.append(detection_info)
    
    print(f"\n[4/4] Generating summary report...")
    
    # Create summary report
    report_path = session_dir / "test_summary.txt"
    with open(report_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("CUSTOM IMAGE TESTING SUMMARY\n")
        f.write("=" * 80 + "\n")
        f.write(f"\nTest Session: {timestamp}\n")
        f.write(f"Model: {MODEL_PATH}\n")
        f.write(f"Confidence Threshold: {confidence_threshold}\n")
        f.write(f"Total Images Tested: {len(test_images)}\n")
        
        total_detections = sum(r['detections'] for r in results_summary)
        images_with_detections = sum(1 for r in results_summary if r['detections'] > 0)
        
        f.write(f"\nTotal Objects Detected: {total_detections}\n")
        f.write(f"Images with Detections: {images_with_detections}/{len(test_images)}\n")
        
        f.write("\n" + "-" * 80 + "\n")
        f.write("DETAILED RESULTS\n")
        f.write("-" * 80 + "\n\n")
        
        for result in results_summary:
            f.write(f"Image: {result['image']}\n")
            f.write(f"  Detections: {result['detections']}\n")
            
            if result['detections'] > 0:
                for cls_name, conf in zip(result['classes'], result['confidences']):
                    f.write(f"    - {cls_name}: {conf:.2%} confidence\n")
            else:
                f.write(f"    No objects detected\n")
            
            f.write(f"  Output: {result['output_path'].name}\n\n")
        
        f.write("=" * 80 + "\n")
        f.write("END OF SUMMARY\n")
        f.write("=" * 80 + "\n")
    
    print("\n" + "=" * 60)
    print("✅ Testing Complete!")
    print("=" * 60)
    print("\nResults saved to: {session_dir}")
    print(f"Summary report: {report_path}")
    print(f"\nTotal detections: {total_detections} objects in {len(test_images)} images")
    print(f"Success rate: {images_with_detections}/{len(test_images)} images with detections")
    
    # Display sample results (first 3 images)
    print(f"\n[Optional] Displaying sample results...")
    display_results(results_summary[:3], session_dir)
    
    return results_summary, session_dir

def display_results(results_summary, session_dir, max_display=3):
    """Display detection results using matplotlib."""
    
    num_display = min(len(results_summary), max_display)
    
    if num_display == 0:
        return
    
    fig, axes = plt.subplots(1, num_display, figsize=(6*num_display, 6))
    
    if num_display == 1:
        axes = [axes]
    
    for idx, (ax, result) in enumerate(zip(axes, results_summary[:num_display])):
        img_path = result['output_path']
        img = cv2.imread(str(img_path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        ax.imshow(img_rgb)
        ax.set_title(f"{result['image']}\n{result['detections']} detection(s)", fontsize=10)
        ax.axis('off')
    
    plt.tight_layout()
    
    # Save figure
    fig_path = session_dir / "results_preview.png"
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"  Preview saved: {fig_path}")
    
    try:
        plt.show()
    except:
        print("  (Cannot display interactive plot in this environment)")
    finally:
        plt.close()

def test_on_validation_set():
    """Test model on the validation set from the dataset."""
    
    print("=" * 60)
    print("Validation Set Testing")
    print("=" * 60)
    
    val_images_dir = BASE_DIR / "yolo_dataset" / "val" / "images"
    
    if not val_images_dir.exists():
        print(f"\n❌ ERROR: Validation set not found!")
        print(f"Expected location: {val_images_dir}")
        return
    
    # Get all validation images
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
    val_images = []
    for ext in image_extensions:
        val_images.extend(list(val_images_dir.glob(ext)))
    
    if not val_images:
        print(f"\n❌ No validation images found!")
        return
    
    print(f"\nTesting on {len(val_images)} validation images...")
    
    # Run testing
    results_summary, session_dir = test_custom_images(
        image_paths=val_images,
        confidence_threshold=0.25
    )
    
    return results_summary

if __name__ == "__main__":
    # Test on custom images
    print("\nChoose testing mode:")
    print("  1. Test on custom images (from 'custom_test_images' folder)")
    print("  2. Test on validation set")
    print("  3. Test on specific image paths")
    
    choice = input("\nEnter choice (1/2/3) [default: 1]: ").strip() or "1"
    
    if choice == "1":
        test_custom_images()
    elif choice == "2":
        test_on_validation_set()
    elif choice == "3":
        # Example: test_custom_images(image_paths=['path/to/image.jpg'])
        print("\nTo test specific images, call from Python:")
        print("  from test_custom_images import test_custom_images")
        print("  test_custom_images(image_paths=['path/to/image1.jpg', 'path/to/image2.jpg'])")
    else:
        print("Invalid choice. Running default (custom images)...")
        test_custom_images()
