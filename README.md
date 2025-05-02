# CPSC_477_project

This is the repository for the end-to-end pipeline to train an adversarial‐RL detector for AI‐generated math solutions, generate solutions on two datasets (MATH and NaturalProofs), and evaluate model performance. Every step is reproducible via SLURM job scripts, or equivalently Python scripts.

## 1. Environment Setup

Make sure you are working in Conda environment and install the required dependencies:

```bash
conda create -n ai-detectors python=3.10 -y
conda activate ai-detectors

pip install \
  torch==2.1.1+cu118 torchvision==0.16.1+cu118 torchaudio==2.1.1+cu118 \
  transformers==4.31.0 datasets==2.14.5 peft==0.5.0 sentence-transformers==2.2.2 \
  scikit-learn==1.3.2 pandas==2.1.1 numpy==2.2.5 matplotlib==3.8.0 accelerate==1.6.0
````

Data files are: 
MATH Dataset: 'dataset.zip'
MATH sample questions and solutions for few-shot prompting: 'example.zip'

---

## 2. MATH Dataset Train and Test Sets Generation (SLURM)
First unzip the MATH Dataset and the sample questions and solutions for few-shot prompting, and then generate baseline solutions for training and different solutions from three different prompts for testing. 

```bash
sbatch run_MATH_train_test.slurm
```

* **Script:** `MATH_train_test.py`


## 3. Training (SLURM)

To train both SFT humanizer and SFT detector models, then perform Adversarial Reinforcement Learning:

```bash
sbatch run_train_RL.slurm
```

* **Output:**
  * `humanizer_sft/` (humanizer SFT checkpoint)
  * `detector_sft/` (detector SFT checkpoint)
  * `saved_detector_rl/` (detector RL checkpoint)

---

## 4. NaturalProofs Solution Generation

Generate proofs on the NaturalProofs dataset via Deepseek under three conditions:

```bash
sbatch run_naturalproofs_inference.slurm
```

* **Script:** `inference_naturalproofs.py`

---

## 5. Detector Inference & Evaluation

### 5.1 Apply Detector to MATH Outputs

```bash
sbatch run_inference.slurm
```

* **Script:** `inference.py`
* **Output:** `detector_inference.npz` (logits/scores for each MATH test set solution from each detector model)

### 5.2 Apply Detector to NaturalProofs Outputs

```bash
sbatch run_eval_naturalproofs.slurm
```

* **Script:** `eval_naturalproofs.py`
* **Output:** `naturalproofs_detector_inference.npz` (logits/scores for each NaturalProofs problem from each detector model)

---

## 6. Analysis & Plotting

Use the Jupyter notebooks for final metrics and visualizations:

```bash
jupyter nbconvert --execute MATH_evaluation_analyses.ipynb
jupyter nbconvert --execute NaturalProofs_evaluation_analyses.ipynb
```

