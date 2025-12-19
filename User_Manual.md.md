# 🧠 Mental Health Sentiment Analysis - User Manual

## Complete Guide for 7-Class Mental Health Classification

---

## 📋 **Table of Contents**

1. [Overview](#overview)
2. [System Requirements](#system-requirements)
3. [Installation & Setup](#installation--setup)
4. [Data Preparation](#data-preparation)
5. [Training the Model](#training-the-model)
6. [Using the Model](#using-the-model)
7. [Understanding Results](#understanding-results)
8. [Troubleshooting](#troubleshooting)
9. [Advanced Usage](#advanced-usage)
10. [FAQ](#faq)

---

## 🎯 **Overview**

This system performs **7-class mental health sentiment analysis** on text statements to classify them into:

| Class | Description |
|-------|-------------|
| **Normal** | Healthy mental state, positive outlook |
| **Depression** | Feelings of sadness, hopelessness, low mood |
| **Suicidal** | Expressions of self-harm or suicide ideation |
| **Anxiety** | Worry, nervousness, panic, fear |
| **Stress** | Overwhelming pressure, tension, strain |
| **Bi-Polar** | Mood swings, extreme emotional changes |
| **Personality Disorder** | Behavioral patterns, relationship issues |

### **Key Features:**
- ✅ **97%+ Accuracy Target**
- ✅ **GPU Acceleration** (RTX 3050 optimized)
- ✅ **Single Statement Testing**
- ✅ **Bulk Dataset Evaluation**
- ✅ **Confidence Percentages**
- ✅ **Interactive Mode**

---

## 💻 **System Requirements**

### **Hardware:**
- **GPU:** NVIDIA GeForce RTX 3050 6GB (recommended)
- **RAM:** 8GB+ system memory
- **Storage:** 5GB free space
- **CPU:** Multi-core processor

### **Software:**
- **OS:** Windows 10/11
- **Python:** 3.11+
- **CUDA:** 12.4+ (for GPU acceleration)

### **Dataset:**
- **Size:** 53,000+ statements
- **Format:** CSV files (no headers)
- **Split:** 70% train, 15% validation, 15% test

---

## 🔧 **Installation & Setup**

### **Step 1: Activate Environment**
```powershell
conda activate torch311
```

### **Step 2: Verify Setup**
```powershell
python quick_start.py
```

**Expected Output:**
```
✅ Python: 3.11.x
✅ PyTorch: 2.5.1
✅ CUDA Available: True
✅ GPU: NVIDIA GeForce RTX 3050 6GB Laptop GPU
✅ All packages installed
```

### **Step 3: Check Data Files**
Ensure these files are in your directory:
- `ml_train.csv`
- `ml_validation.csv`
- `ml_test.csv`

---

## 📊 **Data Preparation**

### **Dataset Format:**
Your CSV files should have **NO HEADERS** and follow this structure:

```
Column 1: Statement (text)
Column 2: Sentiment (class name)
```

### **Example:**
```csv
I feel great today and everything is wonderful!,Normal
I am so depressed and want to give up on life,Depression
I have been feeling anxious about everything lately,Anxiety
The stress is overwhelming and I cannot handle it,Stress
My mood swings are getting worse each day,Bi-Polar
I think I might have a personality disorder,Personality Disorder
I want to end my life and escape this pain,Suicidal
```

### **Data Split:**
- **Training:** 37,100 samples (70%)
- **Validation:** 7,950 samples (15%)
- **Testing:** 7,950 samples (15%)

---

## 🚀 **Training the Model**

### **Method 1: Using Main Application**
```powershell
python main.py
```
1. Select option **1** (Train Model)
2. Wait for training to complete (2-4 hours)
3. Monitor progress and accuracy

### **Method 2: Direct Training**
```powershell
python train_model.py
```

### **Training Process:**
1. **Data Loading:** Loads and preprocesses your dataset
2. **Model Creation:** Initializes BERT-based transformer
3. **Training:** 15 epochs with early stopping
4. **Validation:** Monitors accuracy on validation set
5. **Saving:** Saves best model when target accuracy reached

### **Expected Training Output:**
```
🚀 Starting training for 15 epochs...
🎯 Target accuracy: 97.0%

📊 Epoch 1/15
✅ Train Loss: 1.2345, Train Acc: 0.8234
✅ Val Loss: 1.1234, Val Acc: 0.8567
💾 New best model saved! Accuracy: 0.8567

📊 Epoch 2/15
✅ Train Loss: 0.9876, Train Acc: 0.8765
✅ Val Loss: 0.8765, Val Acc: 0.9123
💾 New best model saved! Accuracy: 0.9123

... (continues until 97%+ accuracy)

🏆 TARGET ACCURACY ACHIEVED! 97.2% >= 97.0%
⏱️ Training completed in: 3.2 hours
```

---

## 🎮 **Using the Model**

### **Method 1: Main Application Interface**
```powershell
python main.py
```

**Main Menu Options:**
1. 🚀 **Train Model** - Train the 7-class classifier
2. 💭 **Test Single Statement** - Test individual statements
3. 📊 **Test Bulk Dataset** - Evaluate entire datasets
4. 🔍 **Check GPU Status** - Verify GPU availability
5. 📁 **Check Data Files** - Verify dataset files
6. ❌ **Exit** - Close application

### **Method 2: Direct Inference**
```powershell
python inference.py
```

---

## 💭 **Single Statement Testing**

### **Interactive Mode:**
1. Run the main application
2. Select option **2** (Test Single Statement)
3. Enter statements when prompted
4. View predictions and confidence scores

### **Example Session:**
```
💭 Enter a statement: I feel really depressed and hopeless

🧠 Mental Health Sentiment Analysis Result
==================================================
📝 Original Text: I feel really depressed and hopeless
🧹 Cleaned Text: i feel really depressed and hopeless
🎯 Predicted Class: Depression
📊 Confidence: 94.23%

📈 All Class Probabilities:
  Normal: 2.15%
  Depression: 94.23%
  Suicidal: 1.87%
  Anxiety: 0.98%
  Stress: 0.45%
  Bi-Polar: 0.22%
  Personality Disorder: 0.10%
```

### **Test Examples:**
Try these example statements:

| Statement | Expected Class |
|-----------|----------------|
| "I feel great today!" | Normal |
| "I'm so depressed" | Depression |
| "I want to hurt myself" | Suicidal |
| "I'm worried about everything" | Anxiety |
| "I'm so stressed out" | Stress |
| "My mood changes constantly" | Bi-Polar |
| "I have trouble with relationships" | Personality Disorder |

---

## 📊 **Bulk Dataset Testing**

### **Testing Process:**
1. Select option **3** (Test Bulk Dataset)
2. System evaluates validation and test sets
3. Generates accuracy reports
4. Saves results to `evaluation_results.csv`

### **Expected Output:**
```
📊 Validation Results:
✅ Accuracy: 0.9723 (97.23%)
📈 Total Samples: 7,950
✅ Correct Predictions: 7,730

📊 Test Results:
✅ Accuracy: 0.9701 (97.01%)
📈 Total Samples: 7,950
✅ Correct Predictions: 7,712

🏆 TARGET ACCURACY ACHIEVED ON TEST SET!
🎯 97.01% >= 97.0%
```

### **Results File:**
The system creates `evaluation_results.csv` with:
- Original text
- True label
- Predicted label
- Confidence percentage

---

## 📈 **Understanding Results**

### **Confidence Scores:**
- **90-100%:** Very confident prediction
- **80-89%:** High confidence
- **70-79%:** Moderate confidence
- **60-69%:** Low confidence
- **<60%:** Very uncertain

### **Class Probabilities:**
Each prediction shows probabilities for all 7 classes:
```
📈 All Class Probabilities:
  Normal: 5.23%
  Depression: 89.45%
  Suicidal: 2.15%
  Anxiety: 1.87%
  Stress: 0.98%
  Bi-Polar: 0.22%
  Personality Disorder: 0.10%
```

### **Accuracy Metrics:**
- **Overall Accuracy:** Percentage of correct predictions
- **Per-Class Accuracy:** Accuracy for each mental health category
- **Confusion Matrix:** Shows prediction patterns
- **Classification Report:** Detailed metrics per class

---

## 🔧 **Troubleshooting**

### **Common Issues:**

#### **1. "Model not found" Error**
```powershell
❌ Trained model not found. Please train the model first.
```
**Solution:** Run training first (option 1 in main menu)

#### **2. "Data files not found" Error**
```powershell
❌ Missing data files: ml_train.csv, ml_validation.csv, ml_test.csv
```
**Solution:** Ensure your dataset files are in the current directory

#### **3. "CUDA not available" Warning**
```powershell
⚠️ No GPU available, using CPU (training will be slower)
```
**Solution:** 
- Check NVIDIA drivers
- Verify CUDA installation
- Restart the application

#### **4. "Out of memory" Error**
```powershell
RuntimeError: CUDA out of memory
```
**Solution:**
- Reduce batch size in training
- Close other GPU applications
- Restart the application

#### **5. Low Accuracy (<97%)**
**Solutions:**
- Increase training epochs
- Check data quality
- Verify class balance
- Try different model parameters

### **Performance Optimization:**

#### **GPU Memory Management:**
```python
# Clear GPU cache
torch.cuda.empty_cache()

# Monitor memory usage
print(f"GPU Memory: {torch.cuda.memory_allocated() / 1024**2:.1f} MB")
```

#### **Batch Size Optimization:**
- **RTX 3050 6GB:** Use batch size 16-32
- **Lower memory:** Use batch size 8-16
- **Higher memory:** Use batch size 32-64

---

## 🚀 **Advanced Usage**

### **Custom Model Training:**
```python
from train_model import MentalHealthTrainer

# Initialize trainer
trainer = MentalHealthTrainer(model_name='bert-base-uncased')

# Custom training parameters
accuracy, predictions, probabilities = trainer.run_complete_training(
    'ml_train.csv', 'ml_validation.csv', 'ml_test.csv', 
    num_epochs=20  # Increase epochs
)
```

### **Custom Inference:**
```python
from inference import MentalHealthInference

# Initialize inference engine
inference = MentalHealthInference()

# Single prediction
result = inference.predict_single("I feel anxious about the future")
print(f"Prediction: {result['class_name']}")
print(f"Confidence: {result['confidence_percentage']:.2f}%")

# Bulk prediction
texts = ["I'm happy", "I'm sad", "I'm worried"]
results = inference.predict_bulk(texts)
```

### **Model Evaluation:**
```python
# Evaluate on custom dataset
results = inference.evaluate_dataset('my_test_data.csv')
print(f"Accuracy: {results['accuracy']:.4f}")
```

---

## ❓ **FAQ**

### **Q: How long does training take?**
**A:** 2-4 hours with RTX 3050 GPU, 8-12 hours with CPU only.

### **Q: What accuracy should I expect?**
**A:** Target is 97%+. With good data, you should achieve 95-98% accuracy.

### **Q: Can I use a different model?**
**A:** Yes, change `model_name` in the code:
- `'bert-base-uncased'` (default)
- `'roberta-base'`
- `'distilbert-base-uncased'`

### **Q: How much GPU memory is needed?**
**A:** 4-5 GB for training, 2-3 GB for inference.

### **Q: Can I add more classes?**
**A:** Yes, modify the `num_classes` parameter and update class mappings.

### **Q: What if my data is imbalanced?**
**A:** The system automatically calculates class weights to handle imbalanced data.

### **Q: Can I save predictions?**
**A:** Yes, bulk evaluation automatically saves results to `evaluation_results.csv`.

### **Q: How do I improve accuracy?**
**A:** 
- Ensure high-quality, balanced data
- Increase training epochs
- Try different model architectures
- Use data augmentation

### **Q: Can I use this for real-time analysis?**
**A:** Yes, the inference engine is optimized for real-time predictions.

### **Q: What if I get errors during training?**
**A:** 
1. Check GPU memory usage
2. Verify data format
3. Ensure all packages are installed
4. Try reducing batch size

---

## 📞 **Support**

### **Getting Help:**
1. **Check this manual** for common solutions
2. **Run diagnostics:** `python quick_start.py`
3. **Check GPU status:** `nvidia-smi`
4. **Verify data format:** Ensure CSV files are correct

### **Performance Tips:**
- Use GPU for training (much faster)
- Close other applications during training
- Monitor GPU memory usage
- Use appropriate batch sizes

### **Best Practices:**
- Always validate your data before training
- Monitor training progress
- Save model checkpoints
- Test on unseen data
- Document your results

---

## 🎉 **Success Indicators**

Your system is working correctly when you see:

✅ **Environment Check:**
```
✅ Python: 3.11.x
✅ PyTorch: 2.5.1
✅ CUDA Available: True
✅ GPU: NVIDIA GeForce RTX 3050 6GB Laptop GPU
```

✅ **Training Success:**
```
🏆 TARGET ACCURACY ACHIEVED! 97.2% >= 97.0%
⏱️ Training completed in: 3.2 hours
```

✅ **Inference Success:**
```
🎯 Predicted Class: Depression
📊 Confidence: 94.23%
```

---

**🎯 You're now ready to use your Mental Health Sentiment Analysis system!**

*Last Updated: October 2024*
*Version: 1.0.0*
