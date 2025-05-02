#!/usr/bin/env python3
# evaluate_and_save_detectors.py

import os
from collections import OrderedDict

import numpy as np
import pandas as pd
import torch
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForSequenceClassification,
)
from peft import PeftModel

def build_aligned_longform(
    baseline_path="test_baseline.csv",
    few_shot_path="test_debug_all.csv",
    harder_path="test_harder_debug_test.csv",
):
    """
    Reads in three CSVs of AI‐generated and human answers,
    aligns them on 'problem', and returns a long‐form DataFrame
    with one row per (question, variant).
    """
    baseline = pd.read_csv(baseline_path)
    few_shot = pd.read_csv(few_shot_path)
    harder   = pd.read_csv(harder_path)

    # drop duplicate (id,problem) in harder
    harder = harder.drop_duplicates(subset=["id","problem"], keep="first")

    baseline = baseline.rename(columns={"generated_answer":"ans_base"})
    few_shot = few_shot.rename(columns=   {"generated_answer":"ans_few"})
    harder   = harder.rename(columns=     {"generated_answer":"ans_hard"})

    merged = (
        few_shot[["problem","id","ans_few"]]
        .merge(baseline[["problem","type","level","ground_truth","ans_base"]],
               on="problem", how="inner")
        .merge(harder[  ["problem","ans_hard"]],
               on="problem", how="inner")
    )
    print(f"✅ Aligned {len(merged)} questions")

    rows=[]
    for _,r in merged.iterrows():
        rows += [
          {"id":r.id, "prompt":r.problem, "variant":"baseline",
           "type":r.type, "level":r.level,
           "text":r.ans_base, "label":0},
          {"id":r.id, "prompt":r.problem, "variant":"few_shot",
           "type":r.type, "level":r.level,
           "text":r.ans_few,  "label":0},
          {"id":r.id, "prompt":r.problem, "variant":"harder",
           "type":r.type, "level":r.level,
           "text":r.ans_hard, "label":0},
          {"id":r.id, "prompt":r.problem, "variant":"human",
           "type":r.type, "level":r.level,
           "text":r.ground_truth,"label":1},
        ]
    df = pd.DataFrame(rows)
    print(f"🎯 Expanded to {len(df)} rows ({len(merged)}×4)")
    return df

def predict_detector(model, texts, tokenizer, device,
                     batch_size=16, max_len=256):
    model.to(device).eval()
    all_p, all_q, all_l = [], [], []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        enc = tokenizer(batch,
                        return_tensors="pt",
                        padding="longest",
                        truncation=True,
                        max_length=max_len).to(device)
        with torch.no_grad():
            logits = model(**enc).logits           # (B,2) FP32
            probs  = torch.softmax(logits,dim=-1)[:,1]
            preds  = (probs>=0.5).long().cpu().numpy()
        all_p.append(preds)
        all_q.append(probs.cpu().numpy())
        all_l.append(logits.cpu().numpy())
        del enc, logits, probs, preds
        torch.cuda.empty_cache()
    return (np.concatenate(all_p),
            np.concatenate(all_q),
            np.concatenate(all_l,axis=0))

if __name__=="__main__":
    # 1) Build / load test set
    test_dfp = build_aligned_longform(
        "test_baseline.csv",
        "test_debug_all.csv",
        "test_harder_debug_test.csv"
    ).dropna(subset=["text"]).reset_index(drop=True)

    texts  = test_dfp["text"].astype(str).tolist()
    labels = test_dfp["label"].astype(int).to_numpy()
    N      = len(texts)

    # 2) Detector checkpoints
    MODEL_ORDER = ["Baseline","SFT‑LoRA","RL‑Detector"]
    DET_CKPTS = OrderedDict([
      ("Baseline",    "Qwen/Qwen2.5-1.5B"),
      ("SFT‑LoRA",    "detector_sft"),
      ("RL‑Detector", "saved_detector_rl"),
    ])

    # 3) shared tokenizer + config
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

    LOAD_KWARGS = dict(
        config            = base_cfg,
        trust_remote_code = True,
        torch_dtype       = torch.float32,
        device_map        = None,
        low_cpu_mem_usage = False,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Running on {device}")

    # 4) Inference loop
    from peft import PeftModel

    preds_dict, probs_dict, logits_dict = {},{},{}
    for name in MODEL_ORDER:
        ck = DET_CKPTS[name]
        print(f"\n→ Loading {name:12s}…", end=" ")

        if name=="Baseline":
            model = AutoModelForSequenceClassification.from_pretrained(ck, **LOAD_KWARGS)

        elif name=="SFT‑LoRA":
            base = AutoModelForSequenceClassification.from_pretrained(
                "Qwen/Qwen2.5-1.5B", **LOAD_KWARGS
            )
            model = PeftModel.from_pretrained(
                base, ck,
                torch_dtype      = torch.float32,
                device_map       = None,
                local_files_only = True
            )

        else:  # RL‑Detector
            model = AutoModelForSequenceClassification.from_pretrained(ck, **LOAD_KWARGS)

        print("✅ ", end="")
        model.to(device)
        print(f"Predicting with {name:12s}…", end=" ")

        p,q,l = predict_detector(model, texts, tokenizer, device)
        preds_dict [name] = p
        probs_dict [name] = q
        logits_dict[name] = l
        print("done")

        del model
        if name=="SFT‑LoRA": del base
        torch.cuda.empty_cache()

    # 5) Report & save
    print(f"\n✅ Evaluated {N} examples:")
    for name in MODEL_ORDER:
        acc = (preds_dict[name][:N]==labels).mean()*100
        print(f"  {name:12s} →  ACC = {acc:5.2f}%")

    os.makedirs("results", exist_ok=True)
    # long‐form CSV
    test_dfp.to_csv("results/test_dfp_longform_labeled.csv", index=False)
    # preds / probs / logits
    np.savez(
      "results/detector_inference.npz",
      **{f"{m}_preds":  preds_dict [m] for m in MODEL_ORDER},
      **{f"{m}_probs":  probs_dict [m] for m in MODEL_ORDER},
      **{f"{m}_logits": logits_dict[m] for m in MODEL_ORDER},
    )
    print("\n🎉 Finished — saved everything under ./results/")