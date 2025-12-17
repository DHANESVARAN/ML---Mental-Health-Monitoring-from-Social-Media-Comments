"""
Test script to verify all imports work correctly
"""

import sys
import os

def test_imports():
    """Test all module imports"""
    print("Testing Mental Health Sentiment Analysis Imports...")
    print("=" * 60)
    
    try:
        # Test data loader
        from data_loader import DataPreprocessor
        print("OK DataPreprocessor imported successfully")
        
        # Test model architecture
        from model_architecture import MentalHealthClassifier
        print("OK MentalHealthClassifier imported successfully")
        
        # Test training
        from train_model import MentalHealthTrainer
        print("OK MentalHealthTrainer imported successfully")
        
        # Test inference
        from inference import MentalHealthInference
        print("OK MentalHealthInference imported successfully")
        
        # Test main app
        from main import MentalHealthApp
        print("OK MentalHealthApp imported successfully")
        
        print("\nAll imports successful!")
        return True
        
    except Exception as e:
        print(f"ERROR Import failed: {str(e)}")
        return False

def test_data_files():
    """Test if data files exist"""
    print("\nTesting Data Files...")
    print("=" * 30)
    
    required_files = ['ml_train.csv', 'ml_validation.csv', 'ml_test.csv']
    all_found = True
    
    for file in required_files:
        if os.path.exists(file):
            size = os.path.getsize(file) / (1024 * 1024)  # MB
            print(f"OK {file} ({size:.1f} MB)")
        else:
            print(f"ERROR {file} - NOT FOUND")
            all_found = False
    
    return all_found

def test_gpu():
    """Test GPU availability"""
    print("\nTesting GPU...")
    print("=" * 20)
    
    try:
        import torch
        print(f"OK PyTorch: {torch.__version__}")
        print(f"OK CUDA Available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print(f"OK GPU: {torch.cuda.get_device_name(0)}")
            print(f"OK CUDA Version: {torch.version.cuda}")
            return True
        else:
            print("WARNING No GPU available")
            return False
            
    except Exception as e:
        print(f"ERROR GPU test failed: {str(e)}")
        return False

def main():
    """Main test function"""
    print("Mental Health Sentiment Analysis - Import Test")
    print("=" * 60)
    
    # Test imports
    imports_ok = test_imports()
    
    # Test data files
    data_ok = test_data_files()
    
    # Test GPU
    gpu_ok = test_gpu()
    
    print("\n" + "=" * 60)
    print("TEST SUMMARY:")
    print(f"Imports: {'PASS' if imports_ok else 'FAIL'}")
    print(f"Data Files: {'PASS' if data_ok else 'FAIL'}")
    print(f"GPU: {'PASS' if gpu_ok else 'WARNING'}")
    
    if imports_ok and data_ok:
        print("\nALL TESTS PASSED!")
        print("Your Mental Health Sentiment Analysis system is ready!")
        print("\nNext steps:")
        print("1. Run: python main.py")
        print("2. Select option 1 to train the model")
        print("3. Wait for training to complete (2-4 hours)")
        print("4. Test with single statements or bulk dataset")
    else:
        print("\nSome tests failed. Please check the issues above.")
    
    return imports_ok and data_ok

if __name__ == "__main__":
    main()
