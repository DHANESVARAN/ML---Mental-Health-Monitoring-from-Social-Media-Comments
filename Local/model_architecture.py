"""
Mental Health Sentiment Analysis - Advanced Transformer Model Architecture
7-Class Classification with 97%+ Accuracy Target
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer, AutoConfig
from transformers import get_linear_schedule_with_warmup
from torch.optim import AdamW
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

class MentalHealthClassifier(nn.Module):
    """Advanced Transformer-based Mental Health Sentiment Classifier"""
    
    def __init__(self, model_name: str = 'bert-base-uncased', num_classes: int = 7, dropout_rate: float = 0.3):
        super(MentalHealthClassifier, self).__init__()
        
        self.num_classes = num_classes
        self.model_name = model_name
        
        # Load pre-trained transformer
        self.transformer = AutoModel.from_pretrained(model_name)
        self.config = AutoConfig.from_pretrained(model_name)
        
        # Get hidden size from transformer
        self.hidden_size = self.config.hidden_size
        
        # Advanced classification head with multiple layers
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(self.hidden_size, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_classes)
        )
        
        # Attention pooling for better representation
        self.attention_pooling = nn.MultiheadAttention(
            embed_dim=self.hidden_size,
            num_heads=8,
            dropout=dropout_rate,
            batch_first=True
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize classifier weights"""
        for module in self.classifier:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0)
    
    def forward(self, input_ids, attention_mask):
        """Forward pass through the model"""
        # Get transformer outputs
        transformer_outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        # Get sequence output
        sequence_output = transformer_outputs.last_hidden_state
        
        # Apply attention pooling
        pooled_output, _ = self.attention_pooling(
            sequence_output, sequence_output, sequence_output,
            key_padding_mask=~attention_mask.bool()
        )
        
        # Global average pooling
        pooled_output = pooled_output.mean(dim=1)
        
        # Classification
        logits = self.classifier(pooled_output)
        
        return logits

class ModelTrainer:
    """Advanced model trainer with GPU optimization"""
    
    def __init__(self, model, device, num_classes=7):
        self.model = model
        self.device = device
        self.num_classes = num_classes
        self.model.to(device)
        print(f"ModelTrainer initialized with {num_classes} classes")
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
    
    def setup_optimizer(self, learning_rate=2e-5, weight_decay=0.01):
        """Setup optimizer with different learning rates for different parts"""
        # Different learning rates for transformer and classifier
        transformer_params = []
        classifier_params = []
        
        for name, param in self.model.named_parameters():
            if 'transformer' in name:
                transformer_params.append(param)
            else:
                classifier_params.append(param)
        
        self.optimizer = AdamW([
            {'params': transformer_params, 'lr': learning_rate},
            {'params': classifier_params, 'lr': learning_rate * 10}
        ], weight_decay=weight_decay)
    
    def setup_scheduler(self, num_training_steps, warmup_steps=500):
        """Setup learning rate scheduler"""
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=num_training_steps
        )
    
    def train_epoch(self, train_loader, class_weights=None):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        correct_predictions = 0
        total_predictions = 0
        
        # Setup loss function with class weights
        if class_weights is not None:
            criterion = nn.CrossEntropyLoss(weight=class_weights.to(self.device))
        else:
            criterion = nn.CrossEntropyLoss()
        
        for batch_idx, batch in enumerate(train_loader):
            # Move to device
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            logits = self.model(input_ids, attention_mask)
            loss = criterion(logits, labels)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # Update weights
            self.optimizer.step()
            self.scheduler.step()
            
            # Statistics
            total_loss += loss.item()
            _, predicted = torch.max(logits, 1)
            correct_predictions += (predicted == labels).sum().item()
            total_predictions += labels.size(0)
            
            # Print progress
            if batch_idx % 100 == 0:
                print(f'Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}')
        
        avg_loss = total_loss / len(train_loader)
        accuracy = correct_predictions / total_predictions
        
        self.train_losses.append(avg_loss)
        self.train_accuracies.append(accuracy)
        
        return avg_loss, accuracy
    
    def validate_epoch(self, val_loader, class_weights=None):
        """Validate for one epoch"""
        self.model.eval()
        total_loss = 0
        correct_predictions = 0
        total_predictions = 0
        
        # Setup loss function
        if class_weights is not None:
            criterion = nn.CrossEntropyLoss(weight=class_weights.to(self.device))
        else:
            criterion = nn.CrossEntropyLoss()
        
        with torch.no_grad():
            for batch in val_loader:
                # Move to device
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # Forward pass
                logits = self.model(input_ids, attention_mask)
                loss = criterion(logits, labels)
                
                # Statistics
                total_loss += loss.item()
                _, predicted = torch.max(logits, 1)
                correct_predictions += (predicted == labels).sum().item()
                total_predictions += labels.size(0)
        
        avg_loss = total_loss / len(val_loader)
        accuracy = correct_predictions / total_predictions
        
        self.val_losses.append(avg_loss)
        self.val_accuracies.append(accuracy)
        
        return avg_loss, accuracy
    
    def train(self, train_loader, val_loader, num_epochs=10, class_weights=None, 
              early_stopping_patience=3, target_accuracy=0.97):
        """Complete training loop with early stopping"""
        print(f"🚀 Starting training for {num_epochs} epochs...")
        print(f"🎯 Target accuracy: {target_accuracy*100:.1f}%")
        
        best_val_accuracy = 0
        patience_counter = 0
        
        for epoch in range(num_epochs):
            print(f"\n📊 Epoch {epoch+1}/{num_epochs}")
            print("-" * 50)
            
            # Train
            train_loss, train_acc = self.train_epoch(train_loader, class_weights)
            print(f"✅ Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            
            # Validate
            val_loss, val_acc = self.validate_epoch(val_loader, class_weights)
            print(f"✅ Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            
            # Check for improvement
            if val_acc > best_val_accuracy:
                best_val_accuracy = val_acc
                patience_counter = 0
                # Save best model
                torch.save(self.model.state_dict(), 'best_mental_health_model.pth')
                print(f"💾 New best model saved! Accuracy: {val_acc:.4f}")
            else:
                patience_counter += 1
            
            # Check if target accuracy reached
            if val_acc >= target_accuracy:
                print(f"🎉 Target accuracy {target_accuracy*100:.1f}% reached!")
                break
            
            # Early stopping
            if patience_counter >= early_stopping_patience:
                print(f"⏹️ Early stopping triggered (patience: {early_stopping_patience})")
                break
        
        print(f"\n🏆 Training completed! Best validation accuracy: {best_val_accuracy:.4f}")
        return best_val_accuracy
    
    def evaluate(self, test_loader, class_names):
        """Comprehensive evaluation"""
        self.model.eval()
        all_predictions = []
        all_labels = []
        all_probabilities = []
        
        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                logits = self.model(input_ids, attention_mask)
                probabilities = F.softmax(logits, dim=1)
                _, predicted = torch.max(logits, 1)
                
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
        
        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_predictions)
        
        print(f"\n📊 Evaluation Results:")
        print(f"✅ Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        
        # Classification report
        print(f"\n📋 Classification Report:")
        report = classification_report(all_labels, all_predictions, 
                                    target_names=class_names, digits=4)
        print(report)
        
        # Confusion matrix
        cm = confusion_matrix(all_labels, all_predictions)
        self.plot_confusion_matrix(cm, class_names)
        
        return accuracy, all_predictions, all_probabilities
    
    def plot_confusion_matrix(self, cm, class_names):
        """Plot confusion matrix"""
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names)
        plt.title('Mental Health Sentiment Classification - Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_training_history(self):
        """Plot training history"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss plot
        ax1.plot(self.train_losses, label='Train Loss')
        ax1.plot(self.val_losses, label='Validation Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Accuracy plot
        ax2.plot(self.train_accuracies, label='Train Accuracy')
        ax2.plot(self.val_accuracies, label='Validation Accuracy')
        ax2.set_title('Training and Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
        plt.show()

def create_model(model_name='bert-base-uncased', num_classes=7):
    """Create and return the model"""
    model = MentalHealthClassifier(model_name=model_name, num_classes=num_classes)
    return model

def test_model_architecture():
    """Test the model architecture"""
    print("🧪 Testing model architecture...")
    
    # Create model
    model = create_model()
    print(f"✅ Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Test forward pass
    batch_size = 2
    seq_length = 128
    input_ids = torch.randint(0, 1000, (batch_size, seq_length))
    attention_mask = torch.ones(batch_size, seq_length)
    
    with torch.no_grad():
        logits = model(input_ids, attention_mask)
        print(f"✅ Forward pass successful. Output shape: {logits.shape}")
    
    print("✅ Model architecture test completed!")

if __name__ == "__main__":
    test_model_architecture()
