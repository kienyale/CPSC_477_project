#!/usr/bin/env python3
"""
Test‐only inference with checkpointing: load DeepSeek‑Math-7B locally and generate
answers for every example in the test split, saving to CSV as you go.
If the script is interrupted, rerunning it will resume where it left off.
"""
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
OUTPUT_CSV = "test_harder_debug_test.csv"

login(token=HF_TOKEN)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("PyTorch:", torch.__version__, " Device:", DEVICE)
for i in range(torch.cuda.device_count()):
    props = torch.cuda.get_device_properties(i)
    used = torch.cuda.memory_allocated(i) / 1e9
    print(f"GPU {i}: {props.name} – {props.total_memory/1e9:.1f} GB total (Used: {used:.1f} GB)")

# --------------------------------------------------------------------------- #
# Load model & tokenizer                                                      #
# --------------------------------------------------------------------------- #
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
tokenizer.pad_token    = tokenizer.eos_token
tokenizer.padding_side = "left"
model.eval()

# --------------------------------------------------------------------------- #
# Load datasets                                                                #
# --------------------------------------------------------------------------- #
features = Features({
    "problem":      Value("string"),
    "level":        Value("string"),
    "type":         Value("string"),
    "solution":     Value("string"),
})
dataset = load_dataset(
    "json",
    data_files={"test": "MATH/test/**/*.json"},
    features=features
)
few_shots = load_dataset(
    "json",
    data_files={"examples": "example/*.json"},
    features=features
)

# --------------------------------------------------------------------------- #
# Build per-type few-shot references                                           #
# --------------------------------------------------------------------------- #
samples = defaultdict(list)
for s in tqdm(few_shots["examples"], desc="Building few-shots"):
    samples[s["type"]].append(
        f"sample question: {s['problem']}\n"
        f"sample solution:  {s['solution']}"
    )

# --------------------------------------------------------------------------- #
# Prepare all test records                                                     #
# --------------------------------------------------------------------------- #
records = []
for idx, rec in enumerate(dataset["test"]):
    records.append({
        "id":           idx,
        "problem":      rec["problem"],
        "type":         rec["type"],
        "level":        rec["level"],
        "reference":    "\n\n".join(samples.get(rec["type"], [])),
        "ground_truth": rec["solution"]
    })

print(f"Prepared {len(records)} test records.")

# --------------------------------------------------------------------------- #
# Load checkpoint of processed test examples                                   #
# --------------------------------------------------------------------------- #
processed = set()
if os.path.exists(OUTPUT_CSV):
    done = pd.read_csv(OUTPUT_CSV, usecols=["id"])
    processed = set(done["id"].tolist())
    print(f"Resuming from {len(processed)} already-processed test rows")

# --------------------------------------------------------------------------- #
# Loop & append each result                                                   #
# --------------------------------------------------------------------------- #
csv_exists = os.path.exists(OUTPUT_CSV)

for rec in tqdm(records, desc="Generating & checkpointing"):
    idx = rec["id"]
    if idx in processed:
        continue

    prompt = f"""
    Analyze and understand the mathematical writing style of these human written solutions: {rec['reference']}
    
    You are a patient math tutor. Using the references above, solve the following problem in a clear, human‑like mathematical writing style by:
    1. Proceeding step‐by‐step with full‐sentence explanations.
    2. Making sure your solution is concise and similar in length to the human written solutions.
    3. Boxing your final answer with \\boxed{{}}.
    
    ### Problem:
    {rec['problem']}
    
    ### Solution:
    """
    
    inputs = tokenizer(prompt, return_tensors="pt", padding=True).to(DEVICE)
    outputs = model.generate(
        **inputs,
        pad_token_id=tokenizer.eos_token_id,
        max_new_tokens=2048
    )
    full = tokenizer.decode(outputs[0], skip_special_tokens=True)
    sol = full.split("### Solution:")[-1].strip()
    sol = re.sub(r'\s*\n\s*', ' ', sol)
    sol = re.sub(r'\s+', ' ', sol)

    # build a one-row DataFrame and append it
    df_row = pd.DataFrame([{
        "id":               idx,
        "problem":          rec["problem"],
        "type":             rec["type"],
        "level":            rec["level"],
        "generated_answer": sol,
        "ground_truth":     rec["ground_truth"]
    }])

    df_row.to_csv(
        OUTPUT_CSV,
        mode="a",
        header=not csv_exists,
        index=False,
        encoding="utf-8"
    )
    csv_exists = True
    processed.add(idx)

print(f"✅ Completed. Total rows in {OUTPUT_CSV}: {len(processed)}")