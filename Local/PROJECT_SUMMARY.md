# Mental Health Monitoring from Social Media Posts — Project Summary

## Overview
A complete pipeline to detect mental health signals in social media posts using PyTorch. The project includes data loading/preprocessing, model architecture, training/validation, checkpointing, and inference utilities.

## Repository Structure (key files)
- `data_loader.py`: Dataset class and `DataPreprocessor` for cleaning/tokenizing text, dataloaders
- `model_architecture.py`: PyTorch model definition
- `train_model.py`: Training loop, validation, checkpointing, metrics
- `main.py`: Entry point for training/evaluation workflows
- `inference.py`: Load best model and run predictions on new text
- `quick_start.py`: Fast setup and sample run
- `ml_train.csv`, `ml_validation.csv`, `ml_test.csv`: Data splits
- `best_mental_health_model.pth`: Best checkpoint saved during training
- `requirements.txt`: Python dependencies
- `User_Manual.md`: Detailed user instructions

## Environment and Setup
- OS: Windows 10
- Python environment: Conda env `torch311` (also a local `.venv` exists)
- GPU used for training (see `GPU_ML_Environment_Setup_Guide.md` and `test_gpu_setup.py`)

Install deps (example):
```bash
conda activate torch311
pip install -r requirements.txt
```

## Data
- Training: `ml_train.csv` (65,585 rows)
- Validation: `ml_validation.csv` (14,038 rows)
- Test: `ml_test.csv` (14,040 rows)

Preprocessing handled by `DataPreprocessor` in `data_loader.py` (tokenization, truncation/padding, batching).

## Model
Defined in `model_architecture.py` (text classifier). Trained with cross-entropy, accuracy metric, and model checkpointing on best validation accuracy.

## Training
- Script: `train_model.py` (invoked via `main.py`)
- Epochs: 15
- Total time: ~55.74 hours (GPU)
- Best validation accuracy: 0.8303
- Best model saved: `best_mental_health_model.pth`

Sample terminal milestones:
- Per-epoch batch logs (e.g., `Batch 2300/2304, Loss: ...`)
- `✅ Train Loss`, `✅ Val Loss`, `Val Acc`
- `💾 New best model saved!` on accuracy improvement
- Final: `🏆 Training completed! Best validation accuracy: 0.8303`

Note on repeated output: After completion, some logs were re-printed due to duplicated logging/print emissions (see Known Issues below). Training itself finished successfully.

## Evaluation
- Best model is reloaded for evaluation: `✅ Best model loaded for evaluation`
- Metrics reported on validation/test as configured in `main.py`/`train_model.py`.

## Inference
- Script: `inference.py`
- Loads `best_mental_health_model.pth`
- Example usage:
```bash
python inference.py --text "I feel overwhelmed lately and can't focus"
```

## How to Reproduce Training
```bash
conda activate torch311
python main.py --mode train \
  --train_csv ml_train.csv \
  --val_csv ml_validation.csv \
  --epochs 15 \
  --save_path best_mental_health_model.pth
```
(Adjust flags to match your `main.py` interface.)

## Known Issues and Remedies
- Repeated terminal output after completion
  - Likely causes:
    - Multiple logging handlers attached (each log prints N times)
    - Training/evaluation function invoked more than once
  - Remedies:
    - Guard entry point with `if __name__ == "__main__":`
    - When configuring logging, ensure handlers are added once (e.g., check `if not logger.handlers:` or use `basicConfig(force=True)`)
    - Ensure there is a single call to the training/evaluation routine

## Results
- Best validation accuracy: 83.03%
- Training stabilized with low training loss; generalization indicated by val accuracy around 0.83

## Next Steps
- Add test-time evaluation/reporting (precision/recall/F1, confusion matrix)
- Perform hyperparameter search (LR, batch size, max_seq_len)
- Consider regularization/early stopping to reduce overfitting gap
- Export model with a simple API/CLI for batch inference

---
Generated on: 2025-11-05

