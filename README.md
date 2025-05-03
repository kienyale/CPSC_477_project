# AI-Generated Math Solution Detector

This project implements a detector model to classify between human-written and AI-generated solutions to mathematical problems. The detector is trained using supervised fine-tuning and can be further improved through reinforcement learning.

## Project Structure

```
.
├── config/
│   └── config.yaml         # Configuration settings
├── data/
│   ├── MATH.zip           # MATH dataset
│   └── example_problems.zip # Example problems
├── src/
│   ├── data/
│   │   ├── data_loader.py  # Dataset loading utilities
│   │   └── data_processor.py # Data processing and preparation
│   ├── models/
│   │   └── detector.py     # Detector model implementation
│   ├── training/
│   │   └── trainer.py      # Training loop and utilities
│   └── utils/
│       ├── logging.py      # Logging configuration
│       └── metrics.py      # Evaluation metrics
├── train_detector.py       # Main training script
└── requirements.txt        # Python dependencies
```

## Setup

1. Create a virtual environment and activate it:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Prepare data:
- Place the MATH dataset in `data/MATH.zip`
- Place example problems in `data/example_problems.zip`

## Configuration

The project uses a YAML configuration file (`config/config.yaml`) to manage all settings:

- Model settings (architecture, tokenizer, etc.)
- Training parameters (learning rate, batch size, etc.)
- Data processing settings
- Paths for data and model artifacts
- Logging configuration

## Training

To train the detector model:

```bash
# Train with LoRA (recommended)
python train_detector.py

# Train without LoRA
python train_detector.py --no-lora
```

The training script:
1. Loads and processes the datasets
2. Creates train/test splits
3. Initializes the model (optionally with LoRA)
4. Trains the model using the configured parameters
5. Saves the best model based on validation performance

## Model Architecture

The detector uses the DeBERTa-v3-large model as the base architecture, with the following modifications:

- Binary classification head for AI/human detection
- Optional LoRA adapters for efficient fine-tuning
- Gradient accumulation for effective batch size scaling
- Linear learning rate warmup

## Evaluation Metrics

The model's performance is evaluated using:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC AUC
- Confusion Matrix Statistics

## Logging

Training progress and results are logged to:
- Console output
- Time-stamped log files in the `logs/` directory

## Requirements

Main dependencies:
- PyTorch
- Transformers
- PEFT (Parameter-Efficient Fine-Tuning)
- scikit-learn
- pandas
- PyYAML

See `requirements.txt` for complete list and versions.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

