# 🧠 GPU-Enabled ML Development Environment Setup Guide

## Mental Health Monitoring from Social Media Posts - ML Project

---

## 📋 **Overview**

This guide documents the complete setup of a GPU-enabled machine learning environment for sentiment analysis and mental health monitoring using your **NVIDIA GeForce RTX 3050 6GB Laptop GPU**.

### ✅ **What's Installed:**
- **Miniconda** - Python package manager
- **Python 3.11** - Latest stable version with CUDA support
- **PyTorch 2.5.1** with CUDA 12.4 support
- **Complete ML Stack** for sentiment analysis
- **GPU Acceleration** for your RTX 3050

---

## 🚀 **Quick Start - Activating Your Environment**

### **After PC Restart (Daily Use):**

1. **Open PowerShell as Administrator** (if needed)
2. **Navigate to your project directory:**
   ```powershell
   cd "D:\PROJECT\Personal Project\ML - Mental Health Monitoring from Social Media Posts\Local"
   ```

3. **Activate your GPU environment:**
   ```powershell
   conda activate torch311
   ```

4. **Verify GPU is working:**
   ```powershell
   python test_gpu_setup.py
   ```

5. **Start your ML development:**
   ```powershell
   python main.py
   ```

### **To Deactivate Environment:**
```powershell
conda deactivate
```

---

## 🔧 **Complete Setup Process (One-Time)**

### **Step 1: Install Miniconda**
```powershell
winget install -e --id Anaconda.Miniconda3 --accept-package-agreements --accept-source-agreements
```

### **Step 2: Initialize Conda**
```powershell
& "$env:USERPROFILE\miniconda3\Scripts\conda.exe" init powershell
```

### **Step 3: Accept Terms of Service**
```powershell
& "$env:USERPROFILE\miniconda3\Scripts\conda.exe" tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
& "$env:USERPROFILE\miniconda3\Scripts\conda.exe" tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
& "$env:USERPROFILE\miniconda3\Scripts\conda.exe" tos accept --override-channels --channel https://repo.anaconda.com/pkgs/msys2
```

### **Step 4: Create Python 3.11 Environment**
```powershell
& "$env:USERPROFILE\miniconda3\Scripts\conda.exe" create -y -n torch311 python=3.11
```

### **Step 5: Install PyTorch with CUDA Support**
```powershell
& "$env:USERPROFILE\miniconda3\Scripts\conda.exe" install -y -n torch311 pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia
```

### **Step 6: Install ML Packages**
```powershell
& "$env:USERPROFILE\miniconda3\Scripts\conda.exe" run -n torch311 pip install pandas scikit-learn matplotlib seaborn nltk textblob transformers accelerate
```

### **Step 7: Verify Installation**
```powershell
& "$env:USERPROFILE\miniconda3\Scripts\conda.exe" run -n torch311 python test_gpu_setup.py
```

---

## 📦 **Installed Packages & Versions**

| Package | Version | Purpose |
|---------|---------|---------|
| **PyTorch** | 2.5.1 | Deep learning framework with CUDA support |
| **CUDA** | 12.4 | GPU acceleration |
| **Pandas** | 2.3.3 | Data manipulation and analysis |
| **NumPy** | 2.0.1 | Numerical computing |
| **Scikit-learn** | 1.7.2 | Traditional machine learning |
| **Transformers** | 4.57.1 | Hugging Face transformer models |
| **Matplotlib** | 3.10.7 | Data visualization |
| **Seaborn** | 0.13.2 | Statistical data visualization |
| **NLTK** | 3.9.2 | Natural language processing |
| **TextBlob** | 0.19.0 | Simple sentiment analysis |
| **Accelerate** | 1.11.0 | GPU optimization |

---

## 🎯 **GPU Specifications & Capabilities**

### **Your Hardware:**
- **GPU:** NVIDIA GeForce RTX 3050 6GB Laptop GPU
- **VRAM:** 6.0 GB
- **CUDA Cores:** 2048
- **Memory Bandwidth:** 192 GB/s

### **Perfect For:**
- ✅ **Sentiment Analysis** with transformer models
- ✅ **Batch processing** of 53k+ samples
- ✅ **Real-time inference** for mental health monitoring
- ✅ **Model training** with proper batching
- ✅ **BERT, RoBERTa, DistilBERT** models
- ✅ **Fine-tuning** pre-trained models

---

## 💻 **Development Workflow**

### **Daily Development Process:**

1. **Start your session:**
   ```powershell
   conda activate torch311
   ```

2. **Load your dataset:**
   ```python
   import pandas as pd
   train_data = pd.read_csv('ml_train.csv')
   val_data = pd.read_csv('ml_validation.csv')
   test_data = pd.read_csv('ml_test.csv')
   ```

3. **Use GPU for model training:**
   ```python
   import torch
   device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
   model = model.to(device)
   ```

4. **Monitor GPU usage:**
   ```python
   print(f"GPU Memory: {torch.cuda.memory_allocated(0) / 1024**2:.1f} MB")
   ```

### **Testing Your Setup:**
```powershell
python test_gpu_setup.py
```

---

## 🔍 **Troubleshooting**

### **Common Issues & Solutions:**

#### **1. "conda command not found"**
```powershell
# Restart PowerShell or run:
& "$env:USERPROFILE\miniconda3\Scripts\conda.exe" init powershell
```

#### **2. "CUDA not available"**
```powershell
# Check NVIDIA drivers:
nvidia-smi

# Reinstall PyTorch with CUDA:
conda activate torch311
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia
```

#### **3. "Environment not found"**
```powershell
# List all environments:
conda env list

# Recreate environment:
conda create -y -n torch311 python=3.11
```

#### **4. "Out of memory" errors**
```python
# Reduce batch size:
batch_size = 16  # instead of 32

# Use gradient accumulation:
accumulation_steps = 4
```

---

## 📊 **Performance Expectations**

### **Your RTX 3050 6GB Performance:**

| Task | Expected Performance |
|------|---------------------|
| **BERT Training** | ~2-4 hours for 53k samples |
| **BERT Inference** | ~1000 samples/second |
| **Data Preprocessing** | ~10k samples/minute |
| **Model Fine-tuning** | ~30-60 minutes |

### **Memory Usage Guidelines:**
- **Batch Size 16:** ~2-3 GB VRAM
- **Batch Size 32:** ~4-5 GB VRAM
- **Batch Size 64:** May cause OOM (Out of Memory)

---

## 🎯 **Next Steps for ML Development**

### **1. Data Exploration:**
```python
import pandas as pd
import matplotlib.pyplot as plt

# Load your dataset
df = pd.read_csv('ml_train.csv')
print(df.head())
print(df.info())
print(df['sentiment'].value_counts())
```

### **2. Simple Models First:**
```python
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer

# Start with traditional ML
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['text'])
y = df['sentiment']
```

### **3. Transformer Models:**
```python
from transformers import AutoTokenizer, AutoModel

# Use pre-trained models
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
model = AutoModel.from_pretrained('bert-base-uncased')
```

### **4. GPU Optimization:**
```python
import torch

# Always use GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Monitor memory
torch.cuda.empty_cache()  # Clear cache when needed
```

---

## 📝 **File Structure**

```
D:\PROJECT\Personal Project\ML - Mental Health Monitoring from Social Media Posts\Local\
├── main.py                          # Your main ML script
├── test_gpu_setup.py               # GPU verification script
├── GPU_ML_Environment_Setup_Guide.md # This documentation
├── ml_train.csv                    # Training data (53k samples)
├── ml_validation.csv               # Validation data
├── ml_test.csv                     # Test data
└── .venv/                         # Virtual environment (backup)
```

---

## 🆘 **Support & Resources**

### **Useful Commands:**
```powershell
# Check GPU status
nvidia-smi

# Check conda environments
conda env list

# Check installed packages
conda list -n torch311

# Update packages
conda update -n torch311 --all
```

### **Documentation Links:**
- [PyTorch CUDA Documentation](https://pytorch.org/docs/stable/cuda.html)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/)
- [Scikit-learn Documentation](https://scikit-learn.org/stable/)
- [Pandas Documentation](https://pandas.pydata.org/docs/)

---

## 🎉 **Success Indicators**

Your setup is working correctly when you see:
```
✅ PyTorch version: 2.5.1
✅ CUDA available: True
✅ CUDA version: 12.4
✅ GPU: NVIDIA GeForce RTX 3050 6GB Laptop GPU
✅ GPU Memory: 6.0 GB
✅ GPU matrix multiplication successful
```

---

## 📞 **Quick Reference Card**

### **Daily Commands:**
```powershell
# Activate environment
conda activate torch311

# Run your ML script
python main.py

# Test GPU setup
python test_gpu_setup.py

# Deactivate when done
conda deactivate
```

### **Emergency Reset:**
```powershell
# If everything breaks, recreate environment:
conda env remove -n torch311
conda create -y -n torch311 python=3.11
conda activate torch311
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia
pip install pandas scikit-learn matplotlib seaborn nltk textblob transformers accelerate
```

---

**🎯 Your GPU-enabled ML environment is ready for mental health monitoring development!**

*Last Updated: October 2024*
*Environment: Windows 11, Python 3.11, PyTorch 2.5.1, CUDA 12.4*
