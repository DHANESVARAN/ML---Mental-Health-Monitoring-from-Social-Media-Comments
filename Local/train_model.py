"""
Mental Health Sentiment Analysis - Training Pipeline
7-Class Classification with 97%+ Accuracy Target
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from transformers import AutoTokenizer
import os
import time
from datetime import datetime
import json
import warnings
warnings.filterwarnings('ignore')

from data_loader import DataPreprocessor
from model_architecture import MentalHealthClassifier, ModelTrainer

class MentalHealthTrainer:
    """Complete training pipeline for mental health sentiment analysis"""
    
    def __init__(self, model_name='bert-base-uncased', device=None):
        self.model_name = model_name
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.preprocessor = DataPreprocessor()
        self.tokenizer = None
        self.model = None
        self.trainer = None
        
        print(f"🖥️ Using device: {self.device}")
        if torch.cuda.is_available():
            print(f"🚀 GPU: {torch.cuda.get_device_name(0)}")
            print(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    def setup_tokenizer(self):
        """Setup tokenizer"""
        print(f"🔤 Loading tokenizer: {self.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        print("✅ Tokenizer loaded successfully")
    
    def load_and_preprocess_data(self, train_path, val_path, test_path):
        """Load and preprocess the dataset"""
        print("📊 Loading and preprocessing data...")
        
        # Load data
        train_df, val_df, test_df = self.preprocessor.load_data(train_path, val_path, test_path)
        
        # Create data loaders
        train_loader, val_loader, test_loader = self.preprocessor.create_data_loaders(
            train_df, val_df, test_df, self.tokenizer, batch_size=16, max_length=512
        )
        
        # Get class weights for imbalanced data
        class_weights = self.preprocessor.get_class_weights(train_df)
        print(f"⚖️ Class weights: {class_weights}")
        
        return train_loader, val_loader, test_loader, class_weights
    
    def create_model(self):
        """Create the model"""
        print(f"Creating model: {self.model_name}")
        # Get actual number of classes from the data
        num_classes = len(self.preprocessor.label_encoder.classes_)
        print(f"Number of classes detected: {num_classes}")
        self.model = MentalHealthClassifier(model_name=self.model_name, num_classes=num_classes)
        self.trainer = ModelTrainer(self.model, self.device, num_classes=num_classes)
        print("Model created successfully")
    
    def train(self, train_loader, val_loader, class_weights, num_epochs=15, target_accuracy=0.97):
        """Train the model"""
        print(f"🚀 Starting training...")
        print(f"🎯 Target accuracy: {target_accuracy*100:.1f}%")
        print(f"📅 Training started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Setup optimizer and scheduler
        self.trainer.setup_optimizer(learning_rate=2e-5, weight_decay=0.01)
        
        # Calculate total training steps
        num_training_steps = len(train_loader) * num_epochs
        self.trainer.setup_scheduler(num_training_steps, warmup_steps=500)
        
        print(f"📊 Training steps: {num_training_steps}")
        print(f"🔥 Warmup steps: 500")
        
        # Start training
        start_time = time.time()
        best_accuracy = self.trainer.train(
            train_loader, val_loader, num_epochs, class_weights, 
            early_stopping_patience=5, target_accuracy=target_accuracy
        )
        end_time = time.time()
        
        training_time = end_time - start_time
        print(f"⏱️ Training completed in: {training_time/3600:.2f} hours")
        
        # Plot training history
        self.trainer.plot_training_history()
        
        return best_accuracy
    
    def evaluate_model(self, test_loader):
        """Evaluate the model"""
        print("📊 Evaluating model...")
        
        # Load best model
        if os.path.exists('best_mental_health_model.pth'):
            self.model.load_state_dict(torch.load('best_mental_health_model.pth'))
            print("✅ Best model loaded for evaluation")
        
        # Get class names
        class_names = self.preprocessor.get_class_names()
        
        # Evaluate
        accuracy, predictions, probabilities = self.trainer.evaluate(test_loader, class_names)
        
        return accuracy, predictions, probabilities
    
    def save_model_info(self, accuracy, training_time):
        """Save model information"""
        model_info = {
            'model_name': self.model_name,
            'accuracy': float(accuracy),
            'training_time_hours': training_time / 3600,
            'device': str(self.device),
            'timestamp': datetime.now().isoformat(),
            'class_names': self.preprocessor.get_class_names(),
            'label_mapping': self.preprocessor.label_mapping
        }
        
        with open('model_info.json', 'w') as f:
            json.dump(model_info, f, indent=2)
        
        print("💾 Model information saved to model_info.json")
    
    def run_complete_training(self, train_path, val_path, test_path, num_epochs=15):
        """Run complete training pipeline"""
        print("🧠 Mental Health Sentiment Analysis - Training Pipeline")
        print("=" * 60)
        
        try:
            # Setup tokenizer
            self.setup_tokenizer()
            
            # Load and preprocess data
            train_loader, val_loader, test_loader, class_weights = self.load_and_preprocess_data(
                train_path, val_path, test_path
            )
            
            # Create model
            self.create_model()
            
            # Train model
            start_time = time.time()
            best_accuracy = self.train(train_loader, val_loader, class_weights, num_epochs)
            training_time = time.time() - start_time
            
            # Evaluate model
            test_accuracy, predictions, probabilities = self.evaluate_model(test_loader)
            
            # Save model info
            self.save_model_info(test_accuracy, training_time)
            
            print(f"\n🎉 Training Pipeline Completed!")
            print(f"✅ Best validation accuracy: {best_accuracy:.4f}")
            print(f"✅ Test accuracy: {test_accuracy:.4f}")
            
            if test_accuracy >= 0.97:
                print(f"🏆 TARGET ACCURACY ACHIEVED! {test_accuracy*100:.2f}% >= 97%")
            else:
                print(f"⚠️ Target accuracy not reached. Current: {test_accuracy*100:.2f}%")
            
            return test_accuracy, predictions, probabilities
            
        except Exception as e:
            print(f"❌ Training failed: {str(e)}")
            raise e

def main():
    """Main training function"""
    print("🧠 Mental Health Sentiment Analysis - 7-Class Classification")
    print("🎯 Target: 97%+ Accuracy")
    print("=" * 60)
    
    # Check GPU availability
    if torch.cuda.is_available():
        print(f"🚀 GPU Available: {torch.cuda.get_device_name(0)}")
        print(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        print("⚠️ No GPU available, using CPU (training will be slower)")
    
    # Initialize trainer
    trainer = MentalHealthTrainer(model_name='bert-base-uncased')
    
    # Define data paths
    train_path = 'ml_train.csv'
    val_path = 'ml_validation.csv'
    test_path = 'ml_test.csv'
    
    # Check if data files exist
    for path in [train_path, val_path, test_path]:
        if not os.path.exists(path):
            print(f"❌ Data file not found: {path}")
            print("Please ensure your dataset files are in the current directory")
            return
    
    # Run training
    try:
        accuracy, predictions, probabilities = trainer.run_complete_training(
            train_path, val_path, test_path, num_epochs=15
        )
        
        print(f"\n🎉 Training completed successfully!")
        print(f"📊 Final accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        
    except Exception as e:
        print(f"❌ Training failed: {str(e)}")
        print("Please check your data files and try again")

if __name__ == "__main__":
    main()
