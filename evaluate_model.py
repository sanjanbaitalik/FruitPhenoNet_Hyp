"""
Model Evaluation Script
Evaluates trained YOLO model and generates comprehensive accuracy report
"""

import os
from pathlib import Path
from ultralytics import YOLO
import torch
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Configuration
BASE_DIR = Path(__file__).parent
DATASET_CONFIG = BASE_DIR / "yolo_dataset" / "data.yaml"
MODEL_PATH = BASE_DIR / "runs" / "train" / "weights" / "best.pt"
REPORT_DIR = BASE_DIR / "evaluation_reports"

def generate_accuracy_report(model, results):
    """Generate comprehensive accuracy report with metrics and visualizations."""
    
    print("\n" + "=" * 60)
    print("Model Evaluation & Accuracy Report")
    print("=" * 60)
    
    # Create report directory
    REPORT_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = REPORT_DIR / f"accuracy_report_{timestamp}.txt"
    
    # Validate on test set
    print("\n[1/4] Running validation on test set...")
    test_results = model.val(data=str(DATASET_CONFIG), split='test')
    
    # Extract metrics
    metrics = {
        'mAP50': test_results.results_dict.get('metrics/mAP50(B)', 0),
        'mAP50-95': test_results.results_dict.get('metrics/mAP50-95(B)', 0),
        'Precision': test_results.results_dict.get('metrics/precision(B)', 0),
        'Recall': test_results.results_dict.get('metrics/recall(B)', 0),
    }
    
    # Calculate F1 score
    if metrics['Precision'] > 0 or metrics['Recall'] > 0:
        metrics['F1_Score'] = 2 * (metrics['Precision'] * metrics['Recall']) / (metrics['Precision'] + metrics['Recall'])
    else:
        metrics['F1_Score'] = 0
    
    print("\n[2/4] Computing class-wise metrics...")
    
    # Get class names
    class_names = model.names
    
    # Create comprehensive report
    print("\n[3/4] Generating report...")
    
    report_content = []
    report_content.append("=" * 80)
    report_content.append("YOLO MODEL EVALUATION REPORT")
    report_content.append("FruitPhenoNet - Bell Pepper Detection")
    report_content.append("=" * 80)
    report_content.append(f"\nReport Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_content.append(f"Model: {MODEL_PATH}")
    report_content.append(f"Dataset: {DATASET_CONFIG}")
    
    report_content.append("\n" + "-" * 80)
    report_content.append("OVERALL METRICS")
    report_content.append("-" * 80)
    report_content.append(f"\nmAP@0.5:       {metrics['mAP50']:.4f} ({metrics['mAP50']*100:.2f}%)")
    report_content.append(f"mAP@0.5:0.95:  {metrics['mAP50-95']:.4f} ({metrics['mAP50-95']*100:.2f}%)")
    report_content.append(f"Precision:     {metrics['Precision']:.4f} ({metrics['Precision']*100:.2f}%)")
    report_content.append(f"Recall:        {metrics['Recall']:.4f} ({metrics['Recall']*100:.2f}%)")
    report_content.append(f"F1 Score:      {metrics['F1_Score']:.4f} ({metrics['F1_Score']*100:.2f}%)")
    
    report_content.append("\n" + "-" * 80)
    report_content.append("METRIC DEFINITIONS")
    report_content.append("-" * 80)
    report_content.append("\nmAP (Mean Average Precision):")
    report_content.append("  - mAP@0.5: Average precision at IoU threshold of 0.5")
    report_content.append("  - mAP@0.5:0.95: Average precision across IoU thresholds from 0.5 to 0.95")
    report_content.append("  - Higher is better (closer to 1.0 or 100%)")
    
    report_content.append("\nPrecision:")
    report_content.append("  - Ratio of correct positive predictions to total positive predictions")
    report_content.append("  - Answers: 'Of all detections, how many were correct?'")
    report_content.append("  - Higher precision = fewer false positives")
    
    report_content.append("\nRecall:")
    report_content.append("  - Ratio of correct positive predictions to all actual positives")
    report_content.append("  - Answers: 'Of all ground truth objects, how many were detected?'")
    report_content.append("  - Higher recall = fewer missed detections")
    
    report_content.append("\nF1 Score:")
    report_content.append("  - Harmonic mean of precision and recall")
    report_content.append("  - Balances both metrics (useful when both are important)")
    
    report_content.append("\n" + "-" * 80)
    report_content.append("CLASSES")
    report_content.append("-" * 80)
    for idx, name in class_names.items():
        report_content.append(f"  Class {idx}: {name}")
    
    report_content.append("\n" + "-" * 80)
    report_content.append("MODEL PERFORMANCE SUMMARY")
    report_content.append("-" * 80)
    
    # Performance interpretation
    map50 = metrics['mAP50']
    if map50 >= 0.9:
        performance = "Excellent"
    elif map50 >= 0.75:
        performance = "Good"
    elif map50 >= 0.5:
        performance = "Fair"
    else:
        performance = "Needs Improvement"
    
    report_content.append(f"\nOverall Performance: {performance}")
    report_content.append(f"\nThe model achieved a mAP@0.5 of {metrics['mAP50']*100:.2f}%, which is considered")
    report_content.append(f"{performance.lower()} for object detection tasks.")
    
    if metrics['Precision'] > metrics['Recall']:
        report_content.append("\nThe model has higher precision than recall, meaning it's conservative")
        report_content.append("in making predictions (fewer false positives, but may miss some objects).")
    elif metrics['Recall'] > metrics['Precision']:
        report_content.append("\nThe model has higher recall than precision, meaning it detects most objects")
        report_content.append("but may have some false positive detections.")
    else:
        report_content.append("\nThe model has balanced precision and recall.")
    
    report_content.append("\n" + "-" * 80)
    report_content.append("TRAINING ARTIFACTS")
    report_content.append("-" * 80)
    report_content.append(f"\nConfusion Matrix: {BASE_DIR / 'runs' / 'train' / 'confusion_matrix.png'}")
    report_content.append(f"Results Plot:     {BASE_DIR / 'runs' / 'train' / 'results.png'}")
    report_content.append(f"PR Curve:         {BASE_DIR / 'runs' / 'train' / 'PR_curve.png'}")
    report_content.append(f"F1 Curve:         {BASE_DIR / 'runs' / 'train' / 'F1_curve.png'}")
    
    report_content.append("\n" + "=" * 80)
    report_content.append("END OF REPORT")
    report_content.append("=" * 80)
    
    # Save report
    report_text = "\n".join(report_content)
    with open(report_path, 'w') as f:
        f.write(report_text)
    
    print(f"\n[4/4] Report saved to: {report_path}")
    
    # Print summary to console
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    for line in report_content:
        if "OVERALL METRICS" in line or "mAP" in line or "Precision" in line or "Recall" in line or "F1 Score" in line:
            print(line)
    
    print("\n" + "=" * 60)
    print("✅ Evaluation Complete!")
    print("=" * 60)
    print(f"\nFull report saved at: {report_path}")
    
    # Create metrics DataFrame for additional analysis
    metrics_df = pd.DataFrame([metrics])
    csv_path = REPORT_DIR / f"metrics_{timestamp}.csv"
    metrics_df.to_csv(csv_path, index=False)
    print(f"Metrics CSV saved at: {csv_path}")
    
    return metrics, report_path

def evaluate_model():
    """Main evaluation function."""
    
    # Check if model exists
    if not MODEL_PATH.exists():
        print(f"\n❌ ERROR: Trained model not found!")
        print(f"Expected location: {MODEL_PATH}")
        print("\nPlease run 'train_yolo.py' first to train the model.")
        return
    
    # Load model
    print("\nLoading model...")
    model = YOLO(str(MODEL_PATH))
    
    # Generate report
    metrics, report_path = generate_accuracy_report(model, None)
    
    return metrics, report_path

if __name__ == "__main__":
    evaluate_model()
