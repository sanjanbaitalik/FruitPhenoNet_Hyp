# FruitPhenoNet - YOLO Model Training Pipeline

Complete pipeline for training YOLO object detection models on bell pepper (Fooled You and Numex) datasets.

## 📁 Project Structure

```
labelled images/
├── Fooled You Labelled Images/     # Fooled You variety labels
├── Numex Labelled Images/          # Numex variety labels
├── prepare_dataset.py              # Dataset preparation script
├── train_yolo.py                   # Model training script
├── evaluate_model.py               # Model evaluation & accuracy report
├── test_custom_images.py           # Custom image testing script
├── requirements.txt                # Python dependencies
├── yolo_dataset/                   # Generated dataset (train/val/test)
├── runs/                           # Training outputs & models
├── evaluation_reports/             # Accuracy reports
├── custom_test_images/             # Place test images here
└── test_results/                   # Testing outputs
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

**Note:** If you have a CUDA-capable GPU, install PyTorch with CUDA support for faster training:

```bash
# For CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### 2. Prepare Dataset

**Important:** Ensure your image files (.jpg, .png) are in the same directories as the label files (.txt), or in companion "Images" folders:
- `Fooled You Images/` (for Fooled You variety)
- `Numex Images/` (for Numex variety)

Run the dataset preparation script:

```bash
python prepare_dataset.py
```

This will:
- Find all labeled images
- Split data into train (70%), validation (20%), and test (10%) sets
- Create YOLO-compatible dataset structure
- Generate `data.yaml` configuration file

### 3. Train Model

```bash
python train_yolo.py
```

Training parameters (can be modified in `train_yolo.py`):
- **Model**: YOLOv8n (nano) - fast and lightweight
  - Options: yolov8n.pt, yolov8s.pt, yolov8m.pt, yolov8l.pt, yolov8x.pt
- **Epochs**: 100
- **Batch size**: 16
- **Image size**: 640x640
- **Device**: Auto-detects GPU/CPU

The trained model will be saved in:
- `runs/train/weights/best.pt` (best performing model)
- `runs/train/weights/last.pt` (last epoch model)

### 4. Evaluate Model & Generate Accuracy Report

```bash
python evaluate_model.py
```

This generates:
- Comprehensive accuracy report with:
  - mAP@0.5 and mAP@0.5:0.95
  - Precision, Recall, F1 Score
  - Metric definitions and interpretations
  - Performance summary
- Report saved in `evaluation_reports/`
- Metrics CSV for further analysis

### 5. Test on Custom Images

#### Option A: Test on Custom Images

1. Place your test images in the `custom_test_images/` folder
2. Run:

```bash
python test_custom_images.py
```

3. Choose option 1 when prompted

#### Option B: Test on Validation Set

```bash
python test_custom_images.py
```

Choose option 2 when prompted to test on the validation set.

#### Option C: Test Specific Images (Python API)

```python
from test_custom_images import test_custom_images

# Test specific images
test_custom_images(
    image_paths=['path/to/image1.jpg', 'path/to/image2.jpg'],
    confidence_threshold=0.25
)
```

Results will be saved in `test_results/test_session_[timestamp]/`

## 📊 Understanding the Metrics

### mAP (Mean Average Precision)
- **mAP@0.5**: Average precision at IoU threshold of 0.5
- **mAP@0.5:0.95**: Average precision across IoU thresholds from 0.5 to 0.95
- **Interpretation**: Higher is better (closer to 1.0 or 100%)

### Precision
- Ratio of correct positive predictions to total positive predictions
- Answers: "Of all detections, how many were correct?"
- Higher precision = fewer false positives

### Recall
- Ratio of correct positive predictions to all actual positives
- Answers: "Of all ground truth objects, how many were detected?"
- Higher recall = fewer missed detections

### F1 Score
- Harmonic mean of precision and recall
- Balances both metrics (useful when both are important)

## 📈 Expected Performance

| Performance Level | mAP@0.5 Range |
|------------------|---------------|
| Excellent        | ≥ 0.90        |
| Good             | 0.75 - 0.89   |
| Fair             | 0.50 - 0.74   |
| Needs Improvement| < 0.50        |

## 🔧 Troubleshooting

### Issue: "No image files found"
**Solution**: Ensure image files (.jpg, .png) are in the same folders as label files or in companion "Images" folders.

### Issue: "CUDA out of memory"
**Solution**: 
- Reduce batch size in `train_yolo.py` (e.g., from 16 to 8 or 4)
- Use a smaller model (e.g., yolov8n.pt instead of yolov8l.pt)

### Issue: Low accuracy
**Solution**:
- Increase training epochs
- Check label quality
- Augment dataset with more diverse images
- Try different model sizes (yolov8s.pt, yolov8m.pt)

### Issue: Training too slow
**Solution**:
- Verify GPU is being used (check output for "GPU available")
- Install CUDA-enabled PyTorch
- Reduce image size or use smaller model

## 🎯 Advanced Configuration

### Modifying Training Parameters

Edit `train_yolo.py` to customize:

```python
MODEL_SIZE = 'yolov8n.pt'  # Model variant
EPOCHS = 100               # Training epochs
BATCH_SIZE = 16           # Batch size
IMAGE_SIZE = 640          # Input image size
```

### Adjusting Data Split

Edit `prepare_dataset.py`:

```python
TRAIN_RATIO = 0.7  # 70% train
VAL_RATIO = 0.2    # 20% validation
TEST_RATIO = 0.1   # 10% test
```

### Confidence Threshold

Adjust detection confidence in `test_custom_images.py`:

```python
test_custom_images(confidence_threshold=0.5)  # Higher = more confident detections
```

## 📝 Output Files

### Training Outputs (`runs/train/`)
- `weights/best.pt` - Best model checkpoint
- `weights/last.pt` - Latest model checkpoint
- `results.png` - Training curves (loss, mAP, etc.)
- `confusion_matrix.png` - Confusion matrix
- `PR_curve.png` - Precision-Recall curve
- `F1_curve.png` - F1 score curve

### Evaluation Reports (`evaluation_reports/`)
- `accuracy_report_[timestamp].txt` - Detailed accuracy report
- `metrics_[timestamp].csv` - Metrics in CSV format

### Test Results (`test_results/`)
- `test_session_[timestamp]/` - Test session folder
  - `result_*.jpg` - Annotated images with detections
  - `test_summary.txt` - Summary of all detections
  - `results_preview.png` - Preview of sample results

## 🤝 Support

For issues or questions, check:
1. Error messages in console output
2. Training logs in `runs/train/`
3. Dataset configuration in `yolo_dataset/data.yaml`

## 📚 Additional Resources

- [Ultralytics YOLOv8 Documentation](https://docs.ultralytics.com/)
- [YOLO Format Guide](https://docs.ultralytics.com/datasets/detect/)
- [PyTorch Installation](https://pytorch.org/get-started/locally/)

## 🔄 Complete Workflow

```
1. prepare_dataset.py  → Creates train/val/test splits
2. train_yolo.py       → Trains YOLO model
3. evaluate_model.py   → Generates accuracy report
4. test_custom_images.py → Tests on custom images
```

---

**Note**: This pipeline is designed for the FruitPhenoNet project with bell pepper detection (Fooled You and Numex varieties). The scripts can be adapted for other object detection tasks by modifying the dataset paths and class names.
