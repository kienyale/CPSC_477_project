# ┌────────────────────────────────────────────────────────────────────────┐
# │ Cell 1: Imports & Setup                                               │
# └────────────────────────────────────────────────────────────────────────┘
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from sklearn.calibration import calibration_curve

import ast
import pandas as pd


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)


# ┌────────────────────────────────────────────────────────────────────────┐
# │ Cell 2: Load Tokenizers & Models                                      │
# └────────────────────────────────────────────────────────────────────────┘
tokenizer_detector = AutoTokenizer.from_pretrained("detector_baseline", trust_remote_code=True)

def build_aligned_longform_from_naturalproofs(
    input_path: str = "LLM_generated_naturalproofs.csv",
) -> pd.DataFrame:
    """
    Reads the CSV from LLM_generated_naturalproofs,
    aligns generated answers once per doc_id,
    and collects all human variants per doc.
    """
    # 1) Load data
    df = pd.read_csv(input_path, index_col=False)

    # 2) Parse composite ID into tuple, extract doc_id
    def parse_id(x):
        if isinstance(x, str):
            tup = tuple(ast.literal_eval(x))
        else:
            tup = tuple(x) if isinstance(x, (list, tuple)) else (x,)
        return tup

    df['ID_parsed'] = df['ID'].apply(parse_id)
    # first element is the document identifier
    df['doc_id'] = df['ID_parsed'].apply(lambda t: t[0])

    # 3) Rename columns
    df = df.rename(columns={
        'Problem': 'prompt',
        'zeroshot': 'ans_base',
        'fewshot': 'ans_few',
        'fewshot2': 'ans_hard',
        'ground_truth': 'ans_human'
    })

    # 4) Drop exact duplicates
    df = df.drop_duplicates(subset=['doc_id','prompt','ans_base','ans_few','ans_hard','ans_human'], keep='first')

    # 5) Group by doc_id + generated answers to collect all human variants
    grouped = (
        df
        .groupby(['doc_id','prompt','ans_base','ans_few','ans_hard'], dropna=False)
        .agg({'ans_human': list})
        .reset_index()
    )

    # 6) Build long form
    rows = []
    for _, r in grouped.iterrows():
        doc = r['doc_id']
        prompt = r['prompt']
        # generated (label=0)
        rows.append({'doc_id': doc, 'prompt': prompt, 'variant': 'baseline',           'text': r['ans_base'], 'label': 0})
        rows.append({'doc_id': doc, 'prompt': prompt, 'variant': 'few_shot',           'text': r['ans_few'],  'label': 0})
        rows.append({'doc_id': doc, 'prompt': prompt, 'variant': 'prompt_engineering','text': r['ans_hard'], 'label': 0})
        # human variants (label=1)
        for human_text in r['ans_human']:
            rows.append({'doc_id': doc, 'prompt': prompt, 'variant': 'human', 'text': human_text, 'label': 1})

    long_df = pd.DataFrame(rows)
    print(f"🎯 Expanded to {len(long_df)} rows ({len(grouped)} docs × (3 generated + N human variants))")
    return long_df


df_long = build_aligned_longform_from_naturalproofs()
df_long.to_csv("aligned_naturalproofs_longform.csv", index=False)


df_long = df_long.dropna(subset=["text"]).reset_index(drop=True)

test_dfp = df_long.copy()


# robust_inference_detectors.py

import transformers, peft
print("transformers:", transformers.__version__)
print("peft:       ", peft.__version__)


import os
from collections import OrderedDict

import torch
import numpy
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForSequenceClassification,
)
from peft import PeftModel

# 1) Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Running on {device}")

# 2) Prepare test set
#    Must define test_dfp with columns ["text","label"] before importing this script
texts  = test_dfp["text"].astype(str).tolist()
labels = test_dfp["label"].astype(int).to_numpy()
N      = len(texts)

# 3) Checkpoint definitions
MODEL_ORDER = ["Baseline", "SFT‑LoRA", "RL‑Detector"]
DET_CKPTS = OrderedDict([
    ("Baseline",    "Qwen/Qwen2.5-1.5B"),
    ("SFT‑LoRA",    "detector_sft"),       # directory with LoRA adapters only
    ("RL‑Detector", "saved_detector_rl"),  # directory with full model weights
])

# 4) Shared tokenizer & base config
tokenizer = AutoTokenizer.from_pretrained(
    "Qwen/Qwen2.5-1.5B", trust_remote_code=True
)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

base_cfg = AutoConfig.from_pretrained(
    "Qwen/Qwen2.5-1.5B", trust_remote_code=True
)
base_cfg.num_labels   = 2
base_cfg.pad_token_id = tokenizer.pad_token_id

# 5) Inference kwargs
#    - Baseline & SFT‑LoRA load base in FP32, then apply LoRA if needed
#    - RL‑Detector load full model in FP32
LOAD_KWARGS = dict(
    config            = base_cfg,
    trust_remote_code = True,
    torch_dtype       = torch.float32,
    device_map        = None,            # force .to(device) below
    low_cpu_mem_usage = False,
)

# 6) Batched inference helper
def predict_detector(model, texts, tokenizer, batch_size=16, max_len=256):
    model.to(device).eval()
    all_preds, all_probs, all_logits = [], [], []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        enc   = tokenizer(
            batch,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=max_len,
        ).to(device)
        with torch.no_grad():
            logits = model(**enc).logits            # (B,2) FP32
            probs  = torch.softmax(logits, dim=-1)[:,1]
            preds  = (probs >= 0.5).long().cpu().numpy()
        all_preds.append(preds)
        all_probs.append(probs.cpu().numpy())
        all_logits.append(logits.cpu().numpy())
        # clean up
        del enc, logits, probs, preds
        torch.cuda.empty_cache()
    return (
        np.concatenate(all_preds,  axis=0),
        np.concatenate(all_probs,  axis=0),
        np.concatenate(all_logits, axis=0),
    )

# 7) Load & evaluate each detector
preds_dict, probs_dict, logits_dict = {}, {}, {}

for name in MODEL_ORDER:
    ckpt = DET_CKPTS[name]
    print(f"\n→ Loading {name:12s}…", end=" ")

    if name == "Baseline":
        model = AutoModelForSequenceClassification.from_pretrained(
            ckpt,
            **LOAD_KWARGS
        )
    elif name == "SFT‑LoRA":
        # load base + attach PEFT adapters
        base = AutoModelForSequenceClassification.from_pretrained(
            "Qwen/Qwen2.5-1.5B",
            **LOAD_KWARGS
        )
        model = PeftModel.from_pretrained(
            base,
            ckpt,
            torch_dtype       = torch.float32,
            device_map        = None,
            local_files_only  = True,
        )
    else:  # RL‑Detector
        model = AutoModelForSequenceClassification.from_pretrained(
            ckpt,
            **LOAD_KWARGS
        )

    print("✅  ", end="")
    # ensure on correct device & dtype
    model.to(device)
    print(f"Predicting with {name:12s}…", end=" ")

    p, q, l = predict_detector(model, texts, tokenizer)
    preds_dict [name] = p
    probs_dict [name] = q
    logits_dict[name] = l

    print("done")

    # teardown
    del model
    if name == "SFT‑LoRA":
        del base
    torch.cuda.empty_cache()

# 8) Summary
print(f"\n✅ Evaluated {N} examples:")
for name in MODEL_ORDER:
    acc = (preds_dict[name][:N] == labels).mean()
    print(f"  {name:12s} → acc = {acc*100:5.2f}%")

# 9) Save outputs for downstream analysis
np.savez(
    "naturalproofs_detector_inference.npz",
    **{f"{m}_preds":  preds_dict[m]  for m in MODEL_ORDER},
    **{f"{m}_probs":  probs_dict[m]  for m in MODEL_ORDER},
    **{f"{m}_logits": logits_dict[m] for m in MODEL_ORDER},
)
print("Saved naturalproofs_detector_inference.npz")