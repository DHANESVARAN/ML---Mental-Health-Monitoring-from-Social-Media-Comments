"""
Mental Health Sentiment Analysis - Data Loading and Preprocessing Module
Handles 7-class classification: Normal, Depression, Suicidal, Anxiety, Stress, Bi-Polar, Personality Disorder
"""

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from transformers import AutoTokenizer
import re
import string
from typing import List, Tuple, Dict
import warnings
warnings.filterwarnings('ignore')

class MentalHealthDataset(Dataset):
    """Custom Dataset for Mental Health Sentiment Analysis"""
    
    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_length: int = 512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        # Tokenize text
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

class DataPreprocessor:
    """Data preprocessing for mental health sentiment analysis"""
    
    def __init__(self):
        self.label_encoder = LabelEncoder()
        self.label_mapping = {
            0: 'Normal',
            1: 'Depression', 
            2: 'Suicidal',
            3: 'Anxiety',
            4: 'Stress',
            5: 'Bi-Polar',
            6: 'Personality Disorder'
        }
        self.reverse_mapping = {v: k for k, v in self.label_mapping.items()}
    
    def clean_text(self, text: str) -> str:
        """Clean and preprocess text data"""
        if pd.isna(text):
            return ""
        
        # Convert to lowercase
        text = str(text).lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove user mentions and hashtags
        text = re.sub(r'@\w+|#\w+', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove punctuation but keep basic sentence structure
        text = text.translate(str.maketrans('', '', string.punctuation))
        
        # Remove numbers
        text = re.sub(r'\d+', '', text)
        
        return text.strip()
    
    def load_data(self, train_path: str, val_path: str, test_path: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Load and preprocess the dataset"""
        print("Loading Mental Health Dataset...")
        
        # Load datasets (no headers)
        train_df = pd.read_csv(train_path, header=None, names=['text', 'sentiment'])
        val_df = pd.read_csv(val_path, header=None, names=['text', 'sentiment'])
        test_df = pd.read_csv(test_path, header=None, names=['text', 'sentiment'])
        
        print(f"Training samples: {len(train_df)}")
        print(f"Validation samples: {len(val_df)}")
        print(f"Test samples: {len(test_df)}")
        
        # Filter out 'Status' class (only 3 samples, not useful for training)
        print("Filtering out 'Status' class (insufficient samples)...")
        train_df = train_df[train_df['sentiment'] != 'Status']
        val_df = val_df[val_df['sentiment'] != 'Status']
        test_df = test_df[test_df['sentiment'] != 'Status']
        
        print(f"After filtering - Training: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
        
        # Clean text data
        print("Cleaning text data...")
        train_df['text'] = train_df['text'].apply(self.clean_text)
        val_df['text'] = val_df['text'].apply(self.clean_text)
        test_df['text'] = test_df['text'].apply(self.clean_text)
        
        # Remove empty texts
        train_df = train_df[train_df['text'].str.len() > 0]
        val_df = val_df[val_df['text'].str.len() > 0]
        test_df = test_df[test_df['text'].str.len() > 0]
        
        # Encode labels
        print("Encoding labels...")
        all_labels = pd.concat([train_df['sentiment'], val_df['sentiment'], test_df['sentiment']])
        self.label_encoder.fit(all_labels)
        
        train_df['encoded_sentiment'] = self.label_encoder.transform(train_df['sentiment'])
        val_df['encoded_sentiment'] = self.label_encoder.transform(val_df['sentiment'])
        test_df['encoded_sentiment'] = self.label_encoder.transform(test_df['sentiment'])
        
        # Display class distribution
        print("\nClass Distribution:")
        for i, class_name in enumerate(self.label_encoder.classes_):
            train_count = (train_df['encoded_sentiment'] == i).sum()
            val_count = (val_df['encoded_sentiment'] == i).sum()
            test_count = (test_df['encoded_sentiment'] == i).sum()
            print(f"  {class_name}: Train={train_count}, Val={val_count}, Test={test_count}")
        
        return train_df, val_df, test_df
    
    def create_data_loaders(self, train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame, 
                          tokenizer, batch_size: int = 16, max_length: int = 512) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Create PyTorch DataLoaders"""
        print("🔄 Creating DataLoaders...")
        
        # Create datasets
        train_dataset = MentalHealthDataset(
            train_df['text'].tolist(),
            train_df['encoded_sentiment'].tolist(),
            tokenizer,
            max_length
        )
        
        val_dataset = MentalHealthDataset(
            val_df['text'].tolist(),
            val_df['encoded_sentiment'].tolist(),
            tokenizer,
            max_length
        )
        
        test_dataset = MentalHealthDataset(
            test_df['text'].tolist(),
            test_df['encoded_sentiment'].tolist(),
            tokenizer,
            max_length
        )
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        
        print(f"✅ DataLoaders created - Batch size: {batch_size}, Max length: {max_length}")
        
        return train_loader, val_loader, test_loader
    
    def get_class_weights(self, train_df: pd.DataFrame) -> torch.Tensor:
        """Calculate class weights for handling imbalanced data"""
        class_counts = train_df['encoded_sentiment'].value_counts().sort_index()
        total_samples = len(train_df)
        num_classes = len(class_counts)  # Use actual number of classes after filtering
        
        # Calculate weights (inverse frequency)
        weights = []
        for i in range(num_classes):
            if i in class_counts.index:
                weight = total_samples / (num_classes * class_counts[i])
            else:
                # If class is missing, use average weight
                weight = 1.0
            weights.append(weight)
        
        print(f"Class weights calculated for {num_classes} classes: {weights}")
        return torch.FloatTensor(weights)
    
    def decode_predictions(self, predictions: np.ndarray) -> List[str]:
        """Decode numerical predictions back to class names"""
        return [self.label_encoder.inverse_transform([pred])[0] for pred in predictions]
    
    def get_class_names(self) -> List[str]:
        """Get list of class names"""
        return list(self.label_encoder.classes_)

def test_data_loading():
    """Test function to verify data loading works correctly"""
    preprocessor = DataPreprocessor()
    
    # Test with sample data
    sample_data = {
        'text': [
            'I feel great today and everything is wonderful!',
            'I am so depressed and want to give up on life',
            'I have been feeling anxious about everything lately',
            'The stress is overwhelming and I cannot handle it',
            'My mood swings are getting worse each day',
            'I think I might have a personality disorder',
            'I want to end my life and escape this pain'
        ],
        'sentiment': [
            'Normal',
            'Depression', 
            'Anxiety',
            'Stress',
            'Bi-Polar',
            'Personality Disorder',
            'Suicidal'
        ]
    }
    
    df = pd.DataFrame(sample_data)
    print("🧪 Testing data preprocessing...")
    print("Original data:")
    print(df)
    
    # Test cleaning
    df['cleaned_text'] = df['text'].apply(preprocessor.clean_text)
    print("\nCleaned data:")
    print(df[['text', 'cleaned_text', 'sentiment']])
    
    print("\n✅ Data preprocessing test completed successfully!")

if __name__ == "__main__":
    test_data_loading()
