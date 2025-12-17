"""
Mental Health Sentiment Analysis - Main Application
7-Class Classification: Normal, Depression, Suicidal, Anxiety, Stress, Bi-Polar, Personality Disorder
Target: 97%+ Accuracy
"""

import torch
import os
import sys
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import our modules
from train_model import MentalHealthTrainer
from inference import MentalHealthInference

class MentalHealthApp:
    """Main application for Mental Health Sentiment Analysis"""
    
    def __init__(self):
        self.app_name = "Mental Health Sentiment Analysis"
        self.version = "1.0.0"
        self.target_accuracy = 0.97
        
        print(f"{self.app_name} v{self.version}")
        print(f"Target Accuracy: {self.target_accuracy*100:.1f}%")
        print("=" * 60)
        
        # Check GPU availability
        self.check_gpu_status()
    
    def check_gpu_status(self):
        """Check GPU availability and status"""
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(f"GPU Available: {gpu_name}")
            print(f"GPU Memory: {gpu_memory:.1f} GB")
        else:
            print("WARNING: No GPU available, using CPU (training will be slower)")
    
    def check_data_files(self):
        """Check if required data files exist"""
        required_files = ['ml_train.csv', 'ml_validation.csv', 'ml_test.csv']
        missing_files = []
        
        for file in required_files:
            if not os.path.exists(file):
                missing_files.append(file)
        
        if missing_files:
            print(f"ERROR: Missing data files: {', '.join(missing_files)}")
            print("Please ensure your dataset files are in the current directory")
            return False
        
        print("OK: All data files found")
        return True
    
    def train_model(self):
        """Train the mental health sentiment analysis model"""
        print("\nStarting Model Training...")
        print("=" * 50)
        
        if not self.check_data_files():
            return False
        
        try:
            # Initialize trainer
            trainer = MentalHealthTrainer(model_name='bert-base-uncased')
            
            # Run training
            accuracy, predictions, probabilities = trainer.run_complete_training(
                'ml_train.csv', 'ml_validation.csv', 'ml_test.csv', num_epochs=15
            )
            
            print(f"\nTraining Completed!")
            print(f"Final Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
            
            if accuracy >= self.target_accuracy:
                print(f"TARGET ACCURACY ACHIEVED! {accuracy*100:.2f}% >= {self.target_accuracy*100:.1f}%")
            else:
                print(f"WARNING: Target accuracy not reached. Current: {accuracy*100:.2f}%")
                print("TIP: Consider increasing training epochs or adjusting model parameters")
            
            return True
            
        except Exception as e:
            print(f"ERROR: Training failed: {str(e)}")
            return False
    
    def test_single_statement(self):
        """Test with a single statement"""
        print("\nSingle Statement Testing")
        print("=" * 50)
        
        if not os.path.exists('best_mental_health_model.pth'):
            print("ERROR: Trained model not found. Please train the model first.")
            return
        
        try:
            # Initialize inference engine
            inference = MentalHealthInference()
            
            # Test examples
            test_statements = [
                "I feel great today and everything is wonderful!",
                "I am so depressed and want to give up on life",
                "I have been feeling anxious about everything lately",
                "The stress is overwhelming and I cannot handle it",
                "My mood swings are getting worse each day",
                "I think I might have a personality disorder",
                "I want to end my life and escape this pain"
            ]
            
            print("Testing with example statements:")
            for i, statement in enumerate(test_statements, 1):
                print(f"\nExample {i}: {statement}")
                result = inference.predict_single(statement)
                inference.print_prediction_summary(result)
            
            # Interactive mode
            print("\nStarting Interactive Mode...")
            print("Type 'quit' to exit")
            inference.interactive_mode()
            
        except Exception as e:
            print(f"ERROR: Single statement testing failed: {str(e)}")
    
    def test_bulk_dataset(self):
        """Test with bulk dataset"""
        print("\nBulk Dataset Testing")
        print("=" * 50)
        
        if not os.path.exists('best_mental_health_model.pth'):
            print("ERROR: Trained model not found. Please train the model first.")
            return
        
        try:
            # Initialize inference engine
            inference = MentalHealthInference()
            
            # Test on validation set
            print("Testing on validation dataset...")
            results = inference.evaluate_dataset('ml_validation.csv')
            
            print(f"\nValidation Results:")
            print(f"Accuracy: {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)")
            print(f"Total Samples: {results['total_samples']}")
            print(f"Correct Predictions: {results['correct_predictions']}")
            
            # Test on test set
            print("\nTesting on test dataset...")
            test_results = inference.evaluate_dataset('ml_test.csv')
            
            print(f"\nTest Results:")
            print(f"Accuracy: {test_results['accuracy']:.4f} ({test_results['accuracy']*100:.2f}%)")
            print(f"Total Samples: {test_results['total_samples']}")
            print(f"Correct Predictions: {test_results['correct_predictions']}")
            
            # Check if target accuracy reached
            if test_results['accuracy'] >= self.target_accuracy:
                print(f"\nTARGET ACCURACY ACHIEVED ON TEST SET!")
                print(f"{test_results['accuracy']*100:.2f}% >= {self.target_accuracy*100:.1f}%")
            else:
                print(f"\nWARNING: Target accuracy not reached on test set")
                print(f"Current: {test_results['accuracy']*100:.2f}% < {self.target_accuracy*100:.1f}%")
            
        except Exception as e:
            print(f"ERROR: Bulk dataset testing failed: {str(e)}")
    
    def show_menu(self):
        """Show main menu"""
        print(f"\n{self.app_name} - Main Menu")
        print("=" * 50)
        print("1. Train Model (7-class classification)")
        print("2. Test Single Statement")
        print("3. Test Bulk Dataset")
        print("4. Check GPU Status")
        print("5. Check Data Files")
        print("6. Exit")
        print("=" * 50)
    
    def run(self):
        """Run the main application"""
        while True:
            self.show_menu()
            
            try:
                choice = input("\nSelect an option (1-6): ").strip()
                
                if choice == '1':
                    self.train_model()
                elif choice == '2':
                    self.test_single_statement()
                elif choice == '3':
                    self.test_bulk_dataset()
                elif choice == '4':
                    self.check_gpu_status()
                elif choice == '5':
                    self.check_data_files()
                elif choice == '6':
                    print("Thank you for using Mental Health Sentiment Analysis!")
                    break
                else:
                    print("WARNING: Invalid choice. Please select 1-6.")
                
                input("\nPress Enter to continue...")
                
            except KeyboardInterrupt:
                print("\n\nApplication interrupted. Goodbye!")
                break
            except Exception as e:
                print(f"ERROR: {str(e)}")
                input("Press Enter to continue...")

def main():
    """Main function"""
    try:
        app = MentalHealthApp()
        app.run()
    except Exception as e:
        print(f"ERROR: Application failed to start: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
