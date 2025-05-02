#!/usr/bin/env python3

import os
import torch
import re
import pandas as pd
from collections import defaultdict
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset, Features, Value
from huggingface_hub import login

HF_TOKEN   = ""
login(token=HF_TOKEN)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Pytorch version is: ", torch.__version__)
print("You are using: ", DEVICE)

"""Expand MATH dataset and the sample questions and solutions for few-shot prompting"""

import zipfile

zip_path = "example.zip"
extract_dir = "./examples"

with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_dir)

zip_path = "dataset.zip"
extract_dir = "./data"

with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_dir)

"""Load DeepSeek and datasets"""

model = AutoModelForCausalLM.from_pretrained(
    "deepseek-ai/deepseek-math-7b-instruct",
    token=HF_TOKEN,
    device_map={"": DEVICE},
    torch_dtype=torch.bfloat16,
    trust_remote_code=True
)


tokenizer = AutoTokenizer.from_pretrained(
    "deepseek-ai/deepseek-math-7b-instruct",
    token=HF_TOKEN
)

features = Features({
    "problem": Value("string"),
    "level": Value("string"),
    "type": Value("string"),
    "solution": Value("string"),
})

dataset = load_dataset(
    "json",
    data_files={
        "train": "data/MATH/train/**/*.json",
        "test": "data/MATH/test/**/*.json"
    },
    features=features
)


few_shots = load_dataset(
    "json",
    data_files={
        "examples": "examples/examples/*.json"
    },
    features=features
)


samples = defaultdict(list)
for sample in tqdm(few_shots['examples']):
    samples[sample['type']].append(f'sample question:{sample["problem"]}, sample solution: {sample["solution"]}')

"""Generate training dataset on MATH Dataset with baseline zero-shot prompting"""

tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

def generate_answer(problem_text):
    prompt = f"""Solve the problem step by step. Use $$ or $$$$ for LaTex. Put the final answer in \\boxed{{}}.


### Problem:
{problem_text}

### Solution:"""
    inputs = tokenizer(prompt, return_tensors="pt",padding=True).to(model.device)
    outputs = model.generate(**inputs,pad_token_id=tokenizer.eos_token_id,  max_new_tokens=2048)
    full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    solution = full_response.split("### Solution:")[-1].strip()
    solution = re.sub(r'\s*\n\s*', ' ', solution)
    solution = re.sub(r'\s+', ' ', solution)
    return solution


results = []
for example in tqdm(dataset["train"]):
    try:
        generated_answer = generate_answer(example["problem"])
        results.append({
            "problem": example["problem"],
            "type": example["type"],
            "level": example["level"],
            "generated_answer": generated_answer,
            "ground_truth": example['solution']
        })
    except Exception as e:
        print(f"Failed on {example['type']} problem: {str(e)}")

results_df = pd.DataFrame(results)
csv_path = "train_baseline.csv"

results_df.to_csv(
    csv_path,
    mode='a',
    index=False,
    header=not os.path.exists(csv_path)
)

"""Three different prompts for MATH test dataset"""

tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left" 

def zeroshot(problem_text):
    prompt = f"""Solve the problem step by step. Use $$ or $$$$ for LaTex. Put the final answer in \\boxed{{}}.

### Problem:
 {problem_text}

### Solution:"""
    inputs = tokenizer(prompt, return_tensors="pt",padding=True).to(model.device)
    outputs = model.generate(**inputs,pad_token_id=tokenizer.eos_token_id,  max_new_tokens=2048)
    full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    solution = full_response.split("### Solution:")[-1].strip()
    return solution

def few_shot(problem_text, reference):
    prompt = f"""Here are some sample questions and sample soutions: {reference}
    Solve the following problem in a similar style to these examples, and put the final answer in \\boxed{{}}.

### Problem:
 {problem_text}

### Solution:"""
    inputs = tokenizer(prompt, return_tensors="pt",padding=True).to(model.device)
    outputs = model.generate(**inputs,pad_token_id=tokenizer.eos_token_id,  max_new_tokens=2048)
    full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    solution = full_response.split("### Solution:")[-1].strip()
    solution = re.sub(r'\s*\n\s*', ' ', solution)
    solution = re.sub(r'\s+', ' ', solution)
    return solution


def fewshot2(problem_text, sample):
    prompt = f"""
    Analyze and understand the mathematical writing style of these human written solutions: {sample}
    
    You are a patient math tutor. Using the references above, solve the following problem in a clear, human‑like mathematical writing style by:
    1. Proceeding step‐by‐step with full‐sentence explanations.
    2. Making sure your solution is concise and similar in length to the human written solutions.
    3. Boxing your final answer with \\boxed{{}}.
    
    ### Problem:
    {problem_text}
    
    ### Solution:
    """

    inputs = tokenizer(prompt, return_tensors="pt",padding=True).to(model.device)
    outputs = model.generate(**inputs,pad_token_id=tokenizer.eos_token_id,  max_new_tokens=2048)
    full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    solution = full_response.split("### Solution:")[-1].strip()
    return solution


results = []
for example in tqdm(dataset["train"]):
    problem = example["Problem"]
    reference = samples[example['type']]
    fs = fewshot(problem, reference)
    zs = zeroshot(problem)
    fs2 = fewshot2(problem, reference)
    print(fs)
    results.append({
        "problem": example["problem"],
        "type": example["type"],
        "level": example["level"],
        "fewshot": fs,
        "zeroshot": zs,
        "fewshot2":fs2,
        "ground_truth": example['Solution']  
    })

results_df = pd.DataFrame(results)
csv_path = "LLM_generated_MATH.csv"

results_df.to_csv(
    csv_path,
    mode='a',      
    index=False,
    header=not os.path.exists(csv_path) 
)
