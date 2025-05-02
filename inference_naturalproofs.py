import torch
import transformers
import ast
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset, DatasetDict, Features, Value
from huggingface_hub import login
import pandas as pd
from tqdm import tqdm
import re
import os

HF_TOKEN=""

login(token=HF_TOKEN)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Pytorch version is: ", torch.__version__)
print("You are using: ", DEVICE)


for i in range(torch.cuda.device_count()):
    mem_total = torch.cuda.get_device_properties(i).total_memory / 1e9 
    mem_used = torch.cuda.memory_allocated(i) / 1e9
    print(f"GPU {i}: {mem_total:.2f} GB (Used: {mem_used:.2f} GB)")


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


NP = load_dataset(
    "wellecks/naturalproofs-gen",
    split="test",
    verification_mode="no_checks"
)

sample = [NP[0]['text'], NP[0]['target'], 
          NP[1]['text'], NP[1]['target']]



tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left" 

def zeroshot(problem_text):
    prompt = f"""Prove the following statement. Format your proof so that it ends with \\n{{qed}}.

### Problem:
 {problem_text}

### Solution:"""
    inputs = tokenizer(prompt, return_tensors="pt",padding=True).to(model.device)
    outputs = model.generate(**inputs,pad_token_id=tokenizer.eos_token_id,  max_new_tokens=2048)
    full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    solution = full_response.split("### Solution:")[-1].strip()
    return solution
    
def fewshot(problem_text, sample):
    prompt = f"""Here are some sample questions and sample human written proofs: {sample}
    Prove the following statement in a similar style to these examples, and format your proof so that it ends with \\n{{qed}}.

### Problem:
 {problem_text}

### Solution:"""
    inputs = tokenizer(prompt, return_tensors="pt",padding=True).to(model.device)
    outputs = model.generate(**inputs,pad_token_id=tokenizer.eos_token_id,  max_new_tokens=2048)
    full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    solution = full_response.split("### Solution:")[-1].strip()
    return solution

def fewshot2(problem_text, sample):
    prompt = f"""
    Analyze and understand the mathematical writing style of these human written proofs: {sample}
    
    You are a patient math tutor. Using the references above, prove the following statement in a clear, human‑like mathematical writing style by:
    1. Proceeding step‐by‐step with full‐sentence explanations.
    2. Making sure your proof is concise and similar in length to the human written proofs.
    3. Format your proof so that it ends with \\n{{qed}}.
    
    ### Problem:
    {problem_text}
    
    ### Solution:
    """

    inputs = tokenizer(prompt, return_tensors="pt",padding=True).to(model.device)
    outputs = model.generate(**inputs,pad_token_id=tokenizer.eos_token_id,  max_new_tokens=2048)
    full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    solution = full_response.split("### Solution:")[-1].strip()
    return solution




output_csv = "LLM_generated_naturalproofs.csv"

# 1) Load existing results (if any)
if os.path.exists(output_csv):
    df_done    = pd.read_csv(output_csv)
    # build set of already-processed (doc_id, sent_idx) tuples
    processed  = set(zip(df_done["doc_id"], df_done["sent_idx"]))
    # carry over prior rows
    results    = df_done.to_dict(orient="records")
else:
    processed = set()
    results   = []

# 2) Iterate only new examples
for example in tqdm(NP, desc="Generating proofs"):
    raw_id            = example["id"]            # e.g. [8570, 0]
    doc_id, sent_idx  = raw_id                   # unpack to ints
    
    # skip if already done
    if (doc_id, sent_idx) in processed:
        continue

    problem = example["text"]
    fs      = fewshot(problem, sample)
    zs      = zeroshot(problem)
    fs2     = fewshot2(problem, sample)

    row = {
        "ID":           raw_id,                # e.g. [8570,0]
        "doc_id":       doc_id,
        "sent_idx":     sent_idx,
        "Problem":      problem,
        "fewshot":      fs,
        "zeroshot":     zs,
        "fewshot2":     fs2,
        "ground_truth": example["target"],
    }

    results.append(row)
    processed.add((doc_id, sent_idx))

    # persist after each new example
    pd.DataFrame(results).to_csv(output_csv, index=False)