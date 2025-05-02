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

## 2. MATH Dataset Zero-Shot Train and Test Sets Generation (SLURM)
First unzip the MATH Dataset and the sample questions and solutions for few-shot prompting, and then generate baseline solutions for both training and testing. 

```bash
sbatch run_MATH_baseline.slurm
```

* **Script:** `MATH_baseline.py`


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

## 4. MATH Test Set Few-Shot Generation

After training, generate few-shot solutions for the MATH test set to make inference harder.

### 4.1 Few-Shot Prompting

```bash
sbatch run_MATH_test_fewshot.slurm
```

* **Script:** `MATH_test_fewshot.py`


### 4.2 Few-Shot + Prompt Engineering

```bash
sbatch run_MATH_test_prompt_engineering.slurm
```

* **Script:** `MATH_test_prompt_engineering.py`

---

## 5. NaturalProofs Solution Generation

Generate proofs on the NaturalProofs dataset via Deepseek under three conditions:

```bash
sbatch run_naturalproofs_inference.slurm
```

* **Script:** `inference_naturalproofs.py`

---

## 6. Detector Inference & Evaluation

### 6.1 Apply Detector to MATH Outputs

```bash
sbatch run_inference.slurm
```

* **Script:** `inference.py`
* **Output:** `detector_inference.npz` (logits/scores for each MATH test set solution from each detector model)

### 6.2 Apply Detector to NaturalProofs Outputs

```bash
sbatch run_eval_naturalproofs.slurm
```

* **Script:** `eval_naturalproofs.py`
* **Output:** `naturalproofs_detector_inference.npz` (logits/scores for each NaturalProof problem from each detector model)

---

## 7. Analysis & Plotting

Use the Jupyter notebooks for final metrics and visualizations:

```bash
jupyter nbconvert --execute MATH_evaluation_analyses.ipynb
jupyter nbconvert --execute NaturalProofs_evaluation_analyses.ipynb
```

