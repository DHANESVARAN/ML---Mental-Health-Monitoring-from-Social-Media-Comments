"""
Mental Health Sentiment Analysis - Inference Module
Single Statement and Bulk Dataset Evaluation
"""

import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
from transformers import AutoTokenizer
import json
import os
from typing import List, Dict, Tuple, Union
import warnings
warnings.filterwarnings('ignore')

from model_architecture import MentalHealthClassifier
from data_loader import DataPreprocessor

class MentalHealthInference:
    """Inference engine for mental health sentiment analysis"""
    
    def __init__(self, model_path='best_mental_health_model.pth', model_name='bert-base-uncased'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_name = model_name
        self.model_path = model_path
        
        # Load model info
        self.model_info = self.load_model_info()
        
        # Initialize components
        self.tokenizer = None
        self.model = None
        self.preprocessor = DataPreprocessor()
        
        # Load model
        self.load_model()
        
        print(f"🧠 Mental Health Sentiment Analysis - Inference Engine")
        print(f"🖥️ Device: {self.device}")
        if torch.cuda.is_available():
            print(f"🚀 GPU: {torch.cuda.get_device_name(0)}")
    
    def load_model_info(self):
        """Load model information"""
        if os.path.exists('model_info.json'):
            with open('model_info.json', 'r') as f:
                return json.load(f)
        else:
            # Default model info
            return {
                'class_names': ['Normal', 'Depression', 'Suicidal', 'Anxiety', 'Stress', 'Bi-Polar', 'Personality Disorder'],
                'label_mapping': {0: 'Normal', 1: 'Depression', 2: 'Suicidal', 3: 'Anxiety', 4: 'Stress', 5: 'Bi-Polar', 6: 'Personality Disorder'}
            }
    
    def load_model(self):
        """Load the trained model"""
        print("🔄 Loading model...")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        # Create model
        self.model = MentalHealthClassifier(model_name=self.model_name, num_classes=7)
        
        # Load trained weights
        if os.path.exists(self.model_path):
            self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
            print("✅ Model weights loaded successfully")
        else:
            print("⚠️ Model weights not found. Using random weights.")
        
        # Move to device
        self.model.to(self.device)
        self.model.eval()
    
    def preprocess_text(self, text: str) -> str:
        """Preprocess single text"""
        return self.preprocessor.clean_text(text)
    
    def predict_single(self, text: str, return_probabilities: bool = True) -> Dict:
        """Predict sentiment for a single statement"""
        # Preprocess text
        cleaned_text = self.preprocess_text(text)
        
        # Tokenize
        encoding = self.tokenizer(
            cleaned_text,
            truncation=True,
            padding='max_length',
            max_length=512,
            return_tensors='pt'
        )
        
        # Move to device
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        
        # Predict
        with torch.no_grad():
            logits = self.model(input_ids, attention_mask)
            probabilities = F.softmax(logits, dim=1)
            predicted_class = torch.argmax(logits, dim=1).item()
            confidence = torch.max(probabilities, dim=1)[0].item()
        
        # Get class name
        class_name = self.model_info['class_names'][predicted_class]
        
        # Prepare result
        result = {
            'text': text,
            'cleaned_text': cleaned_text,
            'predicted_class': predicted_class,
            'class_name': class_name,
            'confidence': confidence,
            'confidence_percentage': confidence * 100
        }
        
        if return_probabilities:
            # Get all class probabilities
            class_probabilities = {}
            for i, class_name in enumerate(self.model_info['class_names']):
                class_probabilities[class_name] = probabilities[0][i].item() * 100
            
            result['class_probabilities'] = class_probabilities
        
        return result
    
    def predict_bulk(self, texts: List[str], return_probabilities: bool = True) -> List[Dict]:
        """Predict sentiment for multiple statements"""
        results = []
        
        print(f"🔄 Processing {len(texts)} statements...")
        
        for i, text in enumerate(texts):
            if i % 100 == 0:
                print(f"📊 Processed {i}/{len(texts)} statements")
            
            result = self.predict_single(text, return_probabilities)
            results.append(result)
        
        print(f"✅ Completed processing {len(texts)} statements")
        return results
    
    def evaluate_dataset(self, dataset_path: str, text_column: str = 'text', 
                        label_column: str = 'sentiment') -> Dict:
        """Evaluate on a dataset file"""
        print(f"📊 Evaluating dataset: {dataset_path}")
        
        # Load dataset
        if dataset_path.endswith('.csv'):
            df = pd.read_csv(dataset_path, header=None, names=[text_column, label_column])
        else:
            df = pd.read_csv(dataset_path)
        
        print(f"📈 Dataset size: {len(df)} samples")
        
        # Get texts and labels
        texts = df[text_column].tolist()
        true_labels = df[label_column].tolist()
        
        # Predict
        predictions = self.predict_bulk(texts, return_probabilities=False)
        
        # Extract predicted classes
        predicted_classes = [pred['class_name'] for pred in predictions]
        
        # Calculate accuracy
        correct = sum(1 for true, pred in zip(true_labels, predicted_classes) if true == pred)
        accuracy = correct / len(true_labels)
        
        # Create results dataframe
        results_df = pd.DataFrame({
            'text': texts,
            'true_label': true_labels,
            'predicted_label': predicted_classes,
            'confidence': [pred['confidence_percentage'] for pred in predictions]
        })
        
        # Save results
        results_df.to_csv('evaluation_results.csv', index=False)
        print("💾 Results saved to evaluation_results.csv")
        
        return {
            'accuracy': accuracy,
            'total_samples': len(true_labels),
            'correct_predictions': correct,
            'results_dataframe': results_df
        }
    
    def get_class_distribution(self, predictions: List[Dict]) -> Dict:
        """Get distribution of predicted classes"""
        class_counts = {}
        for pred in predictions:
            class_name = pred['class_name']
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
        
        total = len(predictions)
        class_distribution = {
            class_name: {
                'count': count,
                'percentage': (count / total) * 100
            }
            for class_name, count in class_counts.items()
        }
        
        return class_distribution
    
    def print_prediction_summary(self, result: Dict):
        """Print a formatted prediction summary"""
        print(f"\n🧠 Mental Health Sentiment Analysis Result")
        print("=" * 50)
        print(f"📝 Original Text: {result['text']}")
        print(f"🧹 Cleaned Text: {result['cleaned_text']}")
        print(f"🎯 Predicted Class: {result['class_name']}")
        print(f"📊 Confidence: {result['confidence_percentage']:.2f}%")
        
        if 'class_probabilities' in result:
            print(f"\n📈 All Class Probabilities:")
            for class_name, prob in result['class_probabilities'].items():
                print(f"  {class_name}: {prob:.2f}%")
    
    def interactive_mode(self):
        """Interactive mode for single statement testing"""
        print("\n🎮 Interactive Mental Health Sentiment Analysis")
        print("Type 'quit' to exit")
        print("=" * 50)
        
        while True:
            text = input("\n💭 Enter a statement: ").strip()
            
            if text.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            
            if not text:
                print("⚠️ Please enter a statement")
                continue
            
            try:
                result = self.predict_single(text)
                self.print_prediction_summary(result)
            except Exception as e:
                print(f"❌ Error: {str(e)}")

def main():
    """Main inference function"""
    print("🧠 Mental Health Sentiment Analysis - Inference Engine")
    print("=" * 60)
    
    # Initialize inference engine
    try:
        inference = MentalHealthInference()
        
        # Check if model exists
        if not os.path.exists('best_mental_health_model.pth'):
            print("⚠️ Trained model not found. Please train the model first.")
            print("Run: python train_model.py")
            return
        
        # Example usage
        print("\n📝 Example Single Statement Prediction:")
        example_text = "I feel really depressed and hopeless about everything"
        result = inference.predict_single(example_text)
        inference.print_prediction_summary(result)
        
        # Interactive mode
        print("\n🎮 Starting Interactive Mode...")
        inference.interactive_mode()
        
    except Exception as e:
        print(f"❌ Error initializing inference engine: {str(e)}")
        print("Please ensure the model is trained first")

if __name__ == "__main__":
    main()
