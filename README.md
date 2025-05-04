# AI-Generated Math Solution Detector

This project implements an adversarial RL-based detector for identifying AI-generated math solutions. The detector is trained on the MATH dataset and evaluated on both MATH and NaturalProofs datasets.

## Project Structure

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

## Installation

1. Clone the repository:
```bash
git clone https://github.com/amyyhwang3/CPSC_477_project.git
cd CPSC_477_project
```

2. Create and activate virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
.\venv\Scripts\activate  # Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Local Development

1. Configure settings in `config/config.yaml`

2. Train the detector:
```bash
python train_detector.py
```

3. Run inference:
```bash
python -m src.models.detector predict \
  --model-path models/detector_rl \
  --input-file data/processed/test.csv \
  --output-file detector_inference.npz
```

### Running on HPC

For large-scale training and evaluation, use the SLURM scripts in `slurm_scripts/`:

```bash
# Generate datasets
sbatch slurm_scripts/run_MATH_train_test.slurm

# Train RL detector
sbatch slurm_scripts/run_train_RL.slurm

# Run evaluations
sbatch slurm_scripts/run_eval_naturalproofs.slurm
```

See `slurm_scripts/README.md` for detailed instructions on running HPC jobs.

## Data

- **MATH Dataset**: Mathematical problems and solutions
- **NaturalProofs**: Mathematical proofs for evaluation
- **Processed Data**: Aligned datasets with human/AI labels

## Models

1. **SFT Detector**: Initial supervised fine-tuned model
2. **RL Detector**: Adversarially trained with RL for robustness

## Results

Analysis notebooks:
- `MATH_evaluation_analyses.ipynb`: MATH dataset results
- `NaturalProofs_evaluation_analyses.ipynb`: NaturalProofs results

## Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/name`)
3. Commit changes (`git commit -am 'Add feature'`)
4. Push branch (`git push origin feature/name`)
5. Create Pull Request

## License

This project is licensed under the MIT License - see LICENSE file for details.

