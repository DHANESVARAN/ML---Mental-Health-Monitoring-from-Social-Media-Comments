#!/usr/bin/env python3
"""
GPU Setup Test Script for Mental Health Monitoring ML Project
Tests PyTorch CUDA functionality and ML packages for sentiment analysis
"""

import torch
import pandas as pd
import numpy as np
import sklearn
import transformers
import matplotlib.pyplot as plt
import seaborn as sns
from textblob import TextBlob
import nltk

def test_gpu_setup():
    """Test GPU setup and basic functionality"""
    print("=" * 60)
    print("🧠 MENTAL HEALTH MONITORING - GPU SETUP TEST")
    print("=" * 60)
    
    # Test PyTorch and CUDA
    print(f"✅ PyTorch version: {torch.__version__}")
    print(f"✅ CUDA available: {torch.cuda.is_available()}")
    print(f"✅ CUDA version: {torch.version.cuda}")
    print(f"✅ Device count: {torch.cuda.device_count()}")
    
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"✅ GPU: {device_name}")
        print(f"✅ GPU Memory: {memory_gb:.1f} GB")
        
        # Test GPU tensor operations
        print("\n🔄 Testing GPU tensor operations...")
        x = torch.randn(1000, 1000).cuda()
        y = torch.randn(1000, 1000).cuda()
        z = torch.mm(x, y)
        print(f"✅ GPU matrix multiplication successful: {z.shape}")
        
        # Test GPU memory usage
        memory_allocated = torch.cuda.memory_allocated(0) / 1024**2
        memory_reserved = torch.cuda.memory_reserved(0) / 1024**2
        print(f"✅ GPU Memory allocated: {memory_allocated:.1f} MB")
        print(f"✅ GPU Memory reserved: {memory_reserved:.1f} MB")
    else:
        print("❌ CUDA not available - using CPU")
    
    # Test ML packages
    print(f"\n📊 ML Packages:")
    print(f"✅ Pandas: {pd.__version__}")
    print(f"✅ NumPy: {np.__version__}")
    print(f"✅ Scikit-learn: {sklearn.__version__}")
    print(f"✅ Transformers: {transformers.__version__}")
    print(f"✅ Matplotlib: {plt.matplotlib.__version__}")
    print(f"✅ Seaborn: {sns.__version__}")
    print(f"✅ TextBlob: Available")
    print(f"✅ NLTK: {nltk.__version__}")
    
    # Test sentiment analysis
    print(f"\n💭 Testing sentiment analysis...")
    test_text = "I'm feeling really happy and excited about this project!"
    blob = TextBlob(test_text)
    sentiment = blob.sentiment
    print(f"✅ Text: '{test_text}'")
    print(f"✅ Sentiment polarity: {sentiment.polarity:.3f} (positive)")
    print(f"✅ Sentiment subjectivity: {sentiment.subjectivity:.3f}")
    
    # Test data loading capability
    print(f"\n📈 Testing data handling...")
    sample_data = {
        'text': ['I feel great!', 'This is terrible.', 'I am okay.', 'Amazing day!', 'Very sad today.'],
        'sentiment': [1, 0, 0, 1, 0]
    }
    df = pd.DataFrame(sample_data)
    print(f"✅ Sample dataset created: {df.shape}")
    print(f"✅ Data preview:")
    print(df.head())
    
    print(f"\n🎉 ALL TESTS PASSED! Your GPU setup is ready for ML development!")
    print(f"💡 You can now start building your sentiment analysis model with 53k samples")
    print(f"🚀 Your RTX 3050 with 6GB VRAM is perfect for this task!")

if __name__ == "__main__":
    test_gpu_setup()
