"""
Mental Health Sentiment Analysis - Quick Start Script
Automated setup and testing for 7-class classification
"""

import os
import sys
import subprocess
import torch
from datetime import datetime

def check_environment():
    """Check if the environment is properly set up"""
    print("Checking Environment...")
    print("=" * 50)
    
    # Check Python version
    python_version = sys.version_info
    print(f"Python: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    # Check PyTorch
    try:
        import torch
        print(f"PyTorch: {torch.__version__}")
        print(f"CUDA Available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"CUDA Version: {torch.version.cuda}")
    except ImportError:
        print("ERROR: PyTorch not installed")
        return False
    
    # Check other packages
    packages = ['pandas', 'numpy', 'sklearn', 'transformers', 'matplotlib', 'seaborn', 'nltk', 'textblob']
    missing_packages = []
    
    for package in packages:
        try:
            __import__(package)
            print(f"OK {package}")
        except ImportError:
            print(f"ERROR {package}")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\nWARNING: Missing packages: {', '.join(missing_packages)}")
        print("Run: pip install -r requirements.txt")
        return False
    
    print("\nEnvironment check completed!")
    return True

def check_data_files():
    """Check if data files exist"""
    print("\nChecking Data Files...")
    print("=" * 50)
    
    required_files = ['ml_train.csv', 'ml_validation.csv', 'ml_test.csv']
    missing_files = []
    
    for file in required_files:
        if os.path.exists(file):
            # Get file size
            size = os.path.getsize(file) / (1024 * 1024)  # MB
            print(f"OK {file} ({size:.1f} MB)")
        else:
            print(f"ERROR {file}")
            missing_files.append(file)
    
    if missing_files:
        print(f"\nWARNING: Missing data files: {', '.join(missing_files)}")
        print("Please ensure your dataset files are in the current directory")
        return False
    
    print("\nAll data files found!")
    return True

def run_quick_test():
    """Run a quick test of the system"""
    print("\nRunning Quick Test...")
    print("=" * 50)
    
    try:
        # Test data loading
        from data_loader import DataPreprocessor
        preprocessor = DataPreprocessor()
        print("OK Data preprocessor loaded")
        
        # Test model architecture
        from model_architecture import MentalHealthClassifier
        model = MentalHealthClassifier()
        print("OK Model architecture loaded")
        
        # Test inference
        from inference import MentalHealthInference
        print("OK Inference engine loaded")
        
        print("\nAll components loaded successfully!")
        return True
        
    except Exception as e:
        print(f"ERROR Quick test failed: {str(e)}")
        return False

def show_next_steps():
    """Show next steps for the user"""
    print("\nNext Steps:")
    print("=" * 50)
    print("1. Prepare your dataset (53k samples, 70% train, 15% val, 15% test)")
    print("2. Run: python main.py")
    print("3. Select option 1 to train the model")
    print("4. Test with single statements (option 2)")
    print("5. Test with bulk dataset (option 3)")
    print("\nExpected Results:")
    print("• Training time: 2-4 hours (with GPU)")
    print("• Target accuracy: 97%+")
    print("• 7-class classification: Normal, Depression, Suicidal, Anxiety, Stress, Bi-Polar, Personality Disorder")

def main():
    """Main function"""
    print("Mental Health Sentiment Analysis - Quick Start")
    print("7-Class Classification with 97%+ Accuracy Target")
    print("=" * 60)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Check environment
    if not check_environment():
        print("\nERROR: Environment check failed. Please fix the issues above.")
        return
    
    # Check data files
    if not check_data_files():
        print("\nERROR: Data files check failed. Please ensure your dataset is ready.")
        return
    
    # Run quick test
    if not run_quick_test():
        print("\nERROR: Quick test failed. Please check your installation.")
        return
    
    # Show next steps
    show_next_steps()
    
    print("\nQuick start completed successfully!")
    print("You're ready to start training your mental health sentiment analysis model!")

if __name__ == "__main__":
    main()
