# YOLO Training Setup Guide

## Prerequisites Check

Before starting, ensure you have:

### ✅ Required Files
- [ ] Label files (.txt) in YOLO format in both directories
- [ ] Corresponding image files (.jpg, .png) with matching names
- [ ] classes.txt files in both label directories

### ✅ System Requirements
- [ ] Python 3.8 or higher
- [ ] At least 4GB RAM (8GB+ recommended)
- [ ] GPU with CUDA support (optional but highly recommended for faster training)
- [ ] 2-5GB free disk space for dataset and models

## Installation Steps

### Step 1: Install Python Packages

Open PowerShell in the project directory and run:

```powershell
pip install -r requirements.txt
```

**For GPU users (CUDA 11.8):**
```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

**For GPU users (CUDA 12.1):**
```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### Step 2: Verify Installation

```powershell
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"
python -c "from ultralytics import YOLO; print('Ultralytics YOLO: OK')"
```

Expected output:
```
PyTorch: 2.x.x
CUDA available: True  (or False if no GPU)
Ultralytics YOLO: OK
```

## Image File Setup

### Important: Image Location

Your label files are in:
- `Fooled You Labelled Images/`
- `Numex Labelled Images/`

You need corresponding image files (.jpg or .png) in ONE of these locations:

**Option 1: Same directory as labels (Recommended)**
```
Fooled You Labelled Images/
  ├── 105_0_0_14-9-22.txt
  ├── 105_0_0_14-9-22.jpg  ← Image here
  ├── 129_0_0_14-9-22.txt
  ├── 129_0_0_14-9-22.jpg  ← Image here
  └── ...
```

**Option 2: Separate Images directory**
```
labelled images/
  ├── Fooled You Labelled Images/    (labels)
  ├── Fooled You Images/             (images)
  ├── Numex Labelled Images/         (labels)
  └── Numex Images/                  (images)
```

### Verify Image-Label Pairing

Run this check in PowerShell:

```powershell
# Check Fooled You directory
Get-ChildItem "Fooled You Labelled Images" -Filter "*.txt" | Select-Object -First 5 | ForEach-Object {
    $base = $_.BaseName
    $img = Get-ChildItem "Fooled You Labelled Images" -Filter "$base.*" | Where-Object {$_.Extension -match '\.(jpg|png)$'}
    Write-Host "$base : " -NoNewline
    if ($img) { Write-Host "✓ Image found" -ForegroundColor Green } else { Write-Host "✗ No image" -ForegroundColor Red }
}
```

## Running the Pipeline

### Method 1: Complete Pipeline (Recommended for First Run)

Run all steps automatically:

```powershell
python run_pipeline.py
```

This will guide you through:
1. Dataset preparation
2. Model training
3. Evaluation & accuracy report
4. Custom image testing

### Method 2: Step-by-Step Execution

#### Step 1: Prepare Dataset
```powershell
python prepare_dataset.py
```

Expected output:
```
[1/5] Collecting Fooled You data...
  Found XX Fooled You samples
[2/5] Collecting Numex data...
  Found XX Numex samples
[3/5] Total samples found: XXX
[4/5] Data split:
  Train: XXX samples
  Val:   XX samples
  Test:  XX samples
[5/5] Copying files...
✅ Dataset preparation complete!
```

#### Step 2: Train Model
```powershell
python train_yolo.py
```

Training will start and show progress:
```
Epoch   GPU_mem   box_loss   cls_loss   dfl_loss   Instances   Size
1/100    2.5G      1.234      0.567      1.123      45         640
...
```

Training time estimates:
- **GPU**: 15-30 minutes for 100 epochs
- **CPU**: 2-4 hours for 100 epochs

#### Step 3: Evaluate Model
```powershell
python evaluate_model.py
```

Generates detailed accuracy report with:
- mAP@0.5 and mAP@0.5:0.95
- Precision, Recall, F1 Score
- Performance interpretation

#### Step 4: Test on Custom Images

**Option A: Prepare test images**
1. Create folder: `custom_test_images/`
2. Add your test images (.jpg, .png)
3. Run:
```powershell
python test_custom_images.py
```

**Option B: Test on validation set**
```powershell
python test_custom_images.py
# Choose option 2
```

## Expected Results

### Dataset Preparation
- Creates `yolo_dataset/` folder with train/val/test splits
- Generates `data.yaml` configuration

### Training Output
- `runs/train/weights/best.pt` - Best model
- `runs/train/weights/last.pt` - Latest model
- Training plots (loss curves, confusion matrix, etc.)

### Evaluation Report
- `evaluation_reports/accuracy_report_[timestamp].txt`
- Detailed metrics and interpretations

### Testing Output
- `test_results/test_session_[timestamp]/`
- Annotated images with bounding boxes
- Summary report

## Troubleshooting

### Issue: "No image files found" during dataset preparation

**Cause**: Image files not in expected locations

**Solution**:
1. Check if images exist:
   ```powershell
   Get-ChildItem "Fooled You Labelled Images" -Filter "*.jpg"
   Get-ChildItem "Fooled You Labelled Images" -Filter "*.png"
   ```
2. Ensure image and label files have matching names
3. Move images to the correct directory

### Issue: "Import ultralytics could not be resolved"

**Cause**: Package not installed

**Solution**:
```powershell
pip install ultralytics
```

### Issue: "CUDA out of memory"

**Cause**: GPU memory insufficient

**Solution**: Edit `train_yolo.py` and reduce batch size:
```python
BATCH_SIZE = 8  # or 4 for very limited memory
```

### Issue: Training is very slow

**Cause**: Using CPU instead of GPU

**Solution**:
1. Check GPU availability:
   ```powershell
   python -c "import torch; print(torch.cuda.is_available())"
   ```
2. Install CUDA-enabled PyTorch (see Step 1)
3. Verify GPU drivers are up to date

### Issue: Low accuracy (<50% mAP)

**Possible Causes & Solutions**:
1. **Insufficient training**: Increase epochs to 150-200
2. **Poor labels**: Review and correct label quality
3. **Small dataset**: Collect more training images
4. **Wrong model**: Try larger model (yolov8s.pt or yolov8m.pt)

## Performance Optimization

### For Faster Training
- Use GPU with CUDA support
- Increase batch size (if memory allows)
- Use smaller image size (e.g., 416 instead of 640)
- Use smaller model (yolov8n.pt is fastest)

### For Better Accuracy
- Use larger model (yolov8m.pt or yolov8l.pt)
- Train for more epochs (150-200)
- Use larger image size (e.g., 1280)
- Ensure high-quality labels

## Next Steps After Training

1. **Review Accuracy Report**
   - Check mAP@0.5 score (target: >0.75 for good performance)
   - Review precision vs recall balance

2. **Analyze Training Plots**
   - Check `runs/train/results.png` for loss curves
   - Look for overfitting (val loss increasing while train loss decreasing)

3. **Test on Real Images**
   - Add your own test images to `custom_test_images/`
   - Run testing script
   - Review detection quality

4. **Fine-tune if Needed**
   - Adjust confidence threshold in testing
   - Retrain with more data if accuracy is low
   - Try different model sizes

## File Structure Reference

```
labelled images/
├── Fooled You Labelled Images/       # Input: labels
├── Numex Labelled Images/            # Input: labels
├── prepare_dataset.py                # Script: dataset prep
├── train_yolo.py                     # Script: training
├── evaluate_model.py                 # Script: evaluation
├── test_custom_images.py             # Script: testing
├── run_pipeline.py                   # Script: complete pipeline
├── requirements.txt                  # Dependencies
├── README.md                         # Documentation
├── SETUP_GUIDE.md                    # This file
├── yolo_dataset/                     # Generated: organized dataset
│   ├── train/
│   ├── val/
│   ├── test/
│   └── data.yaml
├── runs/                             # Generated: training outputs
│   └── train/
│       ├── weights/
│       │   ├── best.pt              # Best model
│       │   └── last.pt              # Last model
│       └── results.png              # Training plots
├── evaluation_reports/               # Generated: accuracy reports
├── custom_test_images/               # Input: your test images
└── test_results/                     # Generated: testing outputs
```

## Quick Commands Summary

```powershell
# Install dependencies
pip install -r requirements.txt

# Complete pipeline
python run_pipeline.py

# Individual steps
python prepare_dataset.py        # 1. Prepare data
python train_yolo.py            # 2. Train model
python evaluate_model.py        # 3. Evaluate & report
python test_custom_images.py    # 4. Test on images

# Check GPU
python -c "import torch; print(torch.cuda.is_available())"
```

## Support & Resources

- **YOLOv8 Documentation**: https://docs.ultralytics.com/
- **PyTorch Installation**: https://pytorch.org/get-started/locally/
- **YOLO Format Guide**: https://docs.ultralytics.com/datasets/detect/

---

**Ready to start? Run:** `python run_pipeline.py`
