# 🤖 AI-Generated Math Solution Detector

A state-of-the-art detector for identifying AI-generated mathematical solutions, trained using Adversarial Reinforcement Learning.

## 📄 Research Paper

[Adversarial Reinforcement Learning based Detection of AI Generated Math Solutions](Adversarial_Reinforcement_Learning_based_Detection_of_AI_Generated_Math_Solutions.pdf)

## 🎯 Key Features

- **Advanced Detection**: Uses adversarial RL to identify AI-generated math solutions with high accuracy
- **Robust Training**: Trained partially on MATH and evaluated on MATH and NaturalProofs datasets for generalizability
- **Dual Model Architecture**: Combines SFT (Supervised Fine-Tuning) and RL-based detection
- **HPC Ready**: Optimized for high-performance computing environments
- **Comprehensive Evaluation**: Extensive testing on multiple mathematical domains

## 🏗️ Project Structure

```
.
├── config/
│   └── config.yaml         # Configuration settings
├── data/
│   ├── MATH/              # MATH dataset
│   └── processed/         # Processed datasets
├── models/
│   ├── detector_rl/       # RL detector checkpoints
│   └── detector_sft/      # SFT detector checkpoints
├── src/
│   ├── data/             # Data loading/processing
│   ├── models/           # Model architectures
│   ├── training/         # Training loops
│   └── utils/            # Helper functions
├── slurm_scripts/        # HPC job scripts
│   └── README.md         # SLURM usage guide
└── requirements.txt      # Python dependencies
```

## 🚀 Quick Start

1. Clone the repository:
```bash
git clone https://github.com/kienyale/CPSC_477_project.git
cd CPSC_477_project
```

2. Set up environment:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
.\venv\Scripts\activate  # Windows

pip install -r requirements.txt
```

3. Configure settings in `config/config.yaml`

4. Run inference:
```bash
python -m src.models.detector predict \
  --model-path models/detector_rl \
  --input-file data/processed/test.csv \
  --output-file detector_inference.npz
```

## 💻 Usage

### Local Development

1. Train the detector:
```bash
python train_detector.py
```

2. Run evaluations:
```bash
python -m src.models.detector evaluate \
  --model-path models/detector_rl \
  --test-file data/processed/test.csv
```

### HPC Deployment

For large-scale training and evaluation, use the SLURM scripts:

```bash
# Generate datasets
sbatch slurm_scripts/run_MATH_train_test.slurm

# Train RL detector
sbatch slurm_scripts/run_train_RL.slurm

# Run evaluations
sbatch slurm_scripts/run_eval_naturalproofs.slurm
```

See `slurm_scripts/README.md` for detailed HPC instructions.

## 📊 Results

Analysis notebooks:
- `MATH_evaluation_analyses.ipynb`: MATH dataset evaluation results
- `NaturalProofs_evaluation_analyses.ipynb`: NaturalProofs evaluation results

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/name`)
3. Commit changes (`git commit -am 'Add feature'`)
4. Push branch (`git push origin feature/name`)
5. Create Pull Request

## 📝 License

This project is licensed under the MIT License - see LICENSE file for details.

