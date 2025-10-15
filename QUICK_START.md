# 🚀 QUICK START GUIDE

## What You Have

A complete YOLO training pipeline for bell pepper detection with:
- ✅ Dataset preparation
- ✅ Model training  
- ✅ Accuracy evaluation & reporting
- ✅ Custom image testing

## ⚡ Quick Start (3 Steps)

### 1️⃣ Install Dependencies (One-time setup)

```powershell
pip install -r requirements.txt
```

**If you have an NVIDIA GPU:**
```powershell
# Install PyTorch with CUDA support for faster training
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### 2️⃣ Ensure Images Are Available

Your label files (.txt) are ready. You need corresponding image files (.jpg or .png).

**Check if images exist:**
```powershell
Get-ChildItem "Fooled You Labelled Images" -Filter "*.jpg" | Select-Object -First 3
```

If no images found, ensure they're in one of these locations:
- Same folder as labels: `Fooled You Labelled Images/*.jpg`
- Separate folder: `Fooled You Images/*.jpg`

### 3️⃣ Run Complete Pipeline

```powershell
python run_pipeline.py
```

This runs everything automatically:
1. Prepares dataset (train/val/test split)
2. Trains YOLO model
3. Generates accuracy report
4. Tests the model

**Expected Time:**
- With GPU: ~20-40 minutes total
- With CPU: ~2-5 hours total

## 📊 What You'll Get

### 1. Trained Model
- Location: `runs/train/weights/best.pt`
- Ready to use for predictions

### 2. Accuracy Report
- Location: `evaluation_reports/accuracy_report_[timestamp].txt`
- Contains:
  - mAP@0.5 (main accuracy metric)
  - Precision & Recall
  - Performance interpretation

### 3. Training Visualizations
- Location: `runs/train/`
- Includes:
  - Loss curves
  - Confusion matrix
  - Precision-Recall curves

### 4. Test Results
- Location: `test_results/`
- Annotated images with detections
- Detection summary report

## 🎯 Understanding Your Results

### Key Metric: mAP@0.5

| Score | Performance | Interpretation |
|-------|-------------|----------------|
| ≥ 0.90 | Excellent | Production-ready |
| 0.75-0.89 | Good | Suitable for most uses |
| 0.50-0.74 | Fair | May need improvement |
| < 0.50 | Poor | Needs more training/data |

### Other Metrics

- **Precision**: Of all detections, how many were correct?
  - High = fewer false alarms
  
- **Recall**: Of all peppers, how many were detected?
  - High = fewer missed detections
  
- **F1 Score**: Balance between precision and recall

## 🧪 Testing on Your Own Images

### After Training:

1. Add your images to `custom_test_images/` folder
2. Run:
   ```powershell
   python test_custom_images.py
   ```
3. Check results in `test_results/` folder

### Using Python:
```python
from test_custom_images import test_custom_images

test_custom_images(
    image_paths=['path/to/your/image.jpg'],
    confidence_threshold=0.25
)
```

## 🔧 Common Issues & Solutions

### ❌ "No image files found"
**Solution**: Place .jpg or .png files alongside .txt label files

### ❌ "CUDA out of memory"
**Solution**: Edit `train_yolo.py`, change `BATCH_SIZE = 8` (or lower)

### ❌ "Import ultralytics could not be resolved"
**Solution**: Run `pip install ultralytics`

### ❌ Training too slow
**Solution**: 
- Install CUDA-enabled PyTorch (see step 1)
- Use smaller model: `MODEL_SIZE = 'yolov8n.pt'` in `train_yolo.py`

### ❌ Low accuracy
**Solution**:
- Train longer: Change `EPOCHS = 200` in `train_yolo.py`
- Use larger model: `MODEL_SIZE = 'yolov8m.pt'`
- Check label quality

## 📁 File Structure

```
📦 labelled images/
├── 📄 run_pipeline.py          ← START HERE (run complete pipeline)
├── 📄 prepare_dataset.py       ← Step 1: Prepare data
├── 📄 train_yolo.py            ← Step 2: Train model
├── 📄 evaluate_model.py        ← Step 3: Get accuracy report
├── 📄 test_custom_images.py    ← Step 4: Test on images
├── 📄 requirements.txt         ← Dependencies
├── 📄 README.md                ← Full documentation
├── 📄 SETUP_GUIDE.md           ← Detailed setup instructions
├── 📄 QUICK_START.md           ← This file
│
├── 📁 Fooled You Labelled Images/  (your labels)
├── 📁 Numex Labelled Images/       (your labels)
│
├── 📁 yolo_dataset/            (generated: organized dataset)
├── 📁 runs/train/              (generated: trained models)
├── 📁 evaluation_reports/      (generated: accuracy reports)
├── 📁 custom_test_images/      (add your test images here)
└── 📁 test_results/            (generated: test outputs)
```

## 🎓 Step-by-Step (Alternative to Pipeline)

If you want more control, run each step separately:

```powershell
# Step 1: Prepare dataset
python prepare_dataset.py

# Step 2: Train model (takes longest)
python train_yolo.py

# Step 3: Evaluate and get accuracy report
python evaluate_model.py

# Step 4: Test on custom images
python test_custom_images.py
```

## 💡 Pro Tips

1. **First run**: Use `python run_pipeline.py` - it's easiest
2. **GPU highly recommended**: 10-20x faster than CPU
3. **Start small**: Default settings work well for most cases
4. **Check accuracy report**: Main metric is mAP@0.5
5. **Test thoroughly**: Try various test images before deployment

## 📞 Need Help?

1. Check `SETUP_GUIDE.md` for detailed troubleshooting
2. Review console output for specific error messages
3. Verify all dependencies installed: `pip list | findstr "ultralytics torch"`

## ✅ Ready? Let's Go!

```powershell
# Install dependencies (if not done)
pip install -r requirements.txt

# Run complete pipeline
python run_pipeline.py
```

That's it! The pipeline will guide you through the rest. 🎉

---

**Expected Timeline:**
- Install: 2-5 minutes
- Dataset prep: 1-2 minutes  
- Training: 15-30 min (GPU) or 2-4 hours (CPU)
- Evaluation: 2-3 minutes
- Testing: 1-2 minutes

**Total: ~30-45 minutes with GPU** or **~3-5 hours with CPU**
