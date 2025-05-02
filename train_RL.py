# environment and logging
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import numpy as np
import pandas as pd
import torch.nn.functional as F
from torch.optim import AdamW
from datasets import Dataset
from sklearn.model_selection import train_test_split
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    DataCollatorWithPadding,
)
from sentence_transformers import SentenceTransformer
from peft import LoraConfig, get_peft_model

DEBUG = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[DEBUG] Using device: {device}, torch {torch.__version__}")

def log_memory_info(ctx=""):
    if DEBUG and device.type == "cuda":
        allocated = torch.cuda.memory_allocated(device) / (1024**2)
        reserved  = torch.cuda.memory_reserved(device) / (1024**2)
        print(f"[MEM] {ctx}: alloc {allocated:.1f}MB, resv {reserved:.1f}MB")


# utility functions
embedder = SentenceTransformer("all-MiniLM-L6-v2", device=device)

def semantic_similarity(a, b):
    # compute normalized cosine similarity between two texts
    e1 = embedder.encode(a, convert_to_tensor=True, show_progress_bar=False)
    e2 = embedder.encode(b, convert_to_tensor=True, show_progress_bar=False)
    return ((F.cosine_similarity(e1, e2, dim=0) + 1) / 2).item()

def compute_length_penalty(txt, ideal_len):
    # penalize generations that exceed the desired length
    return 0.01 * max(0, len(txt.split()) - ideal_len)

def top_p_filtering(logits, top_p=0.9, filter_value=-1e4):
    # apply nucleus (top-p) filtering to logits
    sorted_logits, idx = torch.sort(logits, descending=True)
    probs = torch.softmax(sorted_logits, dim=-1)
    cum_probs = torch.cumsum(probs, dim=-1)
    mask = cum_probs > top_p
    mask[..., 1:] = mask[..., :-1].clone()
    mask[..., 0] = False
    sorted_logits[mask] = filter_value
    return torch.empty_like(logits).scatter(-1, idx, sorted_logits)

def sample_sequence_with_log_probs(model, tok, input_ids, max_len=8, top_p=0.9, window=128):
    # generate a sequence and return cumulative log probability
    model.eval()
    context = input_ids if input_ids.shape[-1] <= window else input_ids[:, -window:]
    generated = context
    log_probs = []
    for _ in range(max_len):
        outputs = model(generated, use_cache=False)
        logits = outputs.logits[:, -1, :].clamp(-1e4, 1e4).nan_to_num(-1e4)
        filtered = top_p_filtering(logits, top_p)
        lps = F.log_softmax(filtered, dim=-1).nan_to_num(-1e4)
        probs = torch.exp(lps)
        zero_mask = probs.sum(dim=1, keepdim=True) <= 0
        if zero_mask.any():
            uniform = torch.ones_like(probs) / probs.size(-1)
            probs = torch.where(zero_mask, uniform, probs)
        probs = probs / probs.sum(dim=1, keepdim=True).clamp(min=1e-6)
        next_token = torch.multinomial(probs, 1)
        token_logp = lps.gather(1, next_token).squeeze(1)
        log_probs.append(token_logp)
        generated = torch.cat([generated, next_token], dim=-1)
        if generated.shape[-1] > window:
            generated = generated[:, -window:]
        if next_token.item() == tok.eos_token_id:
            break
    if not log_probs:
        return generated, torch.zeros(1, device=device)
    return generated, torch.stack(log_probs, dim=1).sum(dim=1)


# data preparation
df = pd.read_csv(
    "train_baseline.csv",
    names=["problem", "type", "level", "ai_solution", "human_solution"],
    header=0
)

train_df = df
ai_sols = train_df["ai_solution"].tolist()
hu_sols = train_df["human_solution"].tolist()
print(f"[DEBUG] Loaded {len(ai_sols)} training examples")


# supervised fine-tuning for humanizer
cfg_h = AutoConfig.from_pretrained("Qwen/Qwen2.5-1.5B", trust_remote_code=True)
cfg_h.use_cache = False

tok_h = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B", trust_remote_code=True)
mdl_h = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-1.5B", config=cfg_h,
    trust_remote_code=True, torch_dtype=torch.float16,
    device_map=None, low_cpu_mem_usage=False
).to(device)

# apply LoRA configuration to humanizer model
lora_cfg = LoraConfig(
    r=8, lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj"],
    lora_dropout=0.1, bias="none"
)
mdl_h = get_peft_model(mdl_h, lora_cfg)

def prep_h(batch):
    # tokenize prompts and reference human solutions
    prompts = ["Rewrite this solution to make it more human: " + a for a in batch["ai_solution"]]
    enc = tok_h(prompts, truncation=True, padding="longest", max_length=128)
    lbl = tok_h(batch["human_solution"], truncation=True, padding="longest", max_length=128)
    enc["labels"] = lbl["input_ids"]
    return enc

from datasets import Dataset as HFDataset
hf_train = HFDataset.from_pandas(train_df)
sft_ds_h = hf_train.map(prep_h, batched=True, remove_columns=train_df.columns.tolist())
sft_ds_h.set_format("torch", ["input_ids", "attention_mask", "labels"])
coll_h = DataCollatorForLanguageModeling(tok_h, mlm=False)

args_h = TrainingArguments(
    output_dir="humanizer_sft",
    num_train_epochs=3,
    per_device_train_batch_size=1,
    learning_rate=5e-5,
    logging_steps=50,
    save_strategy="no"
)
tr_h = Trainer(
    model=mdl_h,
    args=args_h,
    train_dataset=sft_ds_h,
    tokenizer=tok_h,
    data_collator=coll_h,
)
print("[DEBUG] Starting humanizer fine-tuning…")
tr_h.train()
tr_h.save_model("humanizer_sft")
print("[DEBUG] Humanizer fine-tuning complete.")


# supervised fine-tuning for detector
cfg_d = AutoConfig.from_pretrained("Qwen/Qwen2.5-1.5B", trust_remote_code=True)
cfg_d.num_labels = 2
cfg_d.use_cache  = False

tok_d = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B", trust_remote_code=True)
if tok_d.pad_token is None:
    tok_d.pad_token = tok_d.eos_token
cfg_d.pad_token_id = tok_d.pad_token_id

mdl_d = AutoModelForSequenceClassification.from_pretrained(
    "Qwen/Qwen2.5-1.5B",
    config=cfg_d,
    trust_remote_code=True,
    torch_dtype=torch.float32,
    device_map=None,
    low_cpu_mem_usage=False
).to(device)

# freeze base model parameters and enable training of classification head
for p in mdl_d.parameters():
    p.requires_grad = False
for p in mdl_d.score.parameters():
    p.requires_grad = True

# prepare dataset for detector training
det_texts  = ai_sols + hu_sols
det_labels = [0] * len(ai_sols) + [1] * len(hu_sols)
det_ds = Dataset.from_dict({"text": det_texts, "label": det_labels})

def prep_d(batch):
    # tokenize texts and attach labels
    enc = tok_d(batch["text"], truncation=True, padding="longest", max_length=256)
    enc["labels"] = batch["label"]
    return enc

det_ds = det_ds.map(prep_d, batched=True, remove_columns=["text", "label"])
det_ds.set_format("torch", ["input_ids", "attention_mask", "labels"])
coll_d = DataCollatorWithPadding(tok_d)

args_d = TrainingArguments(
    output_dir="detector_sft",
    num_train_epochs=2,
    per_device_train_batch_size=1,
    learning_rate=2e-5,
    logging_steps=20,
    save_strategy="no"
)
tr_d = Trainer(
    model=mdl_d,
    args=args_d,
    train_dataset=det_ds,
    tokenizer=tok_d,
    data_collator=coll_d
)
print("[DEBUG] Starting detector fine-tuning…")
tr_d.train()
tr_d.save_model("detector_sft")
print("[DEBUG] Detector fine-tuning complete.")


# adversarial reinforcement learning loop
optimizer_h = AdamW(mdl_h.parameters(), lr=1e-5)
optimizer_d = AdamW(mdl_d.score.parameters(), lr=1e-5)

num_iters    = 20
batch_size   = 8
gen_max_len  = 16
window_size  = 128
task_prefix  = "Rewrite this solution to make it more human: "

for itr in range(num_iters):
    print(f"\n===== RL Iter {itr+1}/{num_iters} =====")
    mdl_h.train()
    mdl_d.eval()

    rewards, pg_losses = [], []
    adv_texts, adv_labels = [], []

    idxs = np.random.choice(len(ai_sols), batch_size, replace=False)
    for i in idxs:
        ai_txt = ai_sols[i]
        hu_txt = hu_sols[i]

        # generate humanized solution sample
        enc_h = tok_h(task_prefix + ai_txt, return_tensors="pt",
                      truncation=True, padding="longest", max_length=128).to(device)
        gen_ids, logp = sample_sequence_with_log_probs(
            mdl_h, tok_h, enc_h.input_ids,
            max_len=gen_max_len, top_p=0.9, window=window_size
        )
        gen_txt = tok_h.decode(gen_ids[0], skip_special_tokens=True)

        # evaluate generated sample with detector
        det_in = tok_d(gen_txt, return_tensors="pt",
                       truncation=True, padding="longest", max_length=256).to(device)
        with torch.no_grad():
            logits_d = mdl_d(**det_in).logits
        p_human = F.softmax(logits_d, dim=-1)[0, 1].item()

        # compute reward based on detection score, semantic similarity, and length penalty
        sim   = semantic_similarity(gen_txt, hu_txt)
        pen   = compute_length_penalty(gen_txt, len(hu_txt.split()))
        raw_R = p_human - (1 - sim) - pen
        R     = float(np.clip(raw_R, -1.0, 1.0))
        rewards.append(R)

        # update humanizer model with policy gradient
        loss_pg = -(logp * R).mean()
        optimizer_h.zero_grad()
        loss_pg.backward()
        optimizer_h.step()
        pg_losses.append(loss_pg.item())

        # accumulate adversarial examples for detector training
        if np.random.rand() < 0.5:
            adv_texts += [hu_txt, gen_txt]
            adv_labels += [1, 0]
        else:
            adv_texts += [gen_txt, hu_txt]
            adv_labels += [0, 1]

    print(f" humanizer ▶ avg_reward={np.mean(rewards):.4f}, avg_pg_loss={np.mean(pg_losses):.4f}")

    # train detector on adversarial examples
    mdl_d.train()
    batch = tok_d(adv_texts, truncation=True, padding="longest",
                  max_length=256, return_tensors="pt").to(device)
    labels_t = torch.tensor(adv_labels, device=device)
    logits_all = mdl_d(**batch).logits
    loss_d = F.cross_entropy(logits_all, labels_t)

    optimizer_d.zero_grad()
    loss_d.backward()
    optimizer_d.step()

    print(f" detector ▶ adv_loss={loss_d.item():.4f}")

print("\n✅ adversarial reinforcement learning complete.")


# verification: save, reload, and compare detector model
import os
import torch
import numpy as np
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from collections import OrderedDict

# assume detector model, tokenizer, and test data are available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE_DIR = "saved_detector_rl"
os.makedirs(SAVE_DIR, exist_ok=True)

# save current detector model and tokenizer
print(f"[INFO] Saving in-memory detector to '{SAVE_DIR}' …")
mdl_d.save_pretrained(SAVE_DIR)
tok_d.save_pretrained(SAVE_DIR)

# create sample batch with four AI and four human solutions
sample_texts = (
    test_df["ai_solution"].tolist()[:4] +
    test_df["human_solution"].tolist()[:4]
)
enc = tok_d(
    sample_texts,
    return_tensors="pt",
    padding=True,
    truncation=True,
    max_length=128
).to(device)

# obtain logits from original model
mdl_d.eval()
with torch.no_grad():
    orig_logits = mdl_d(**enc).logits.detach().cpu()

# record parameters of the original model
orig_state = OrderedDict()
for k, v in mdl_d.state_dict().items():
    if v.device.type != "meta":
        orig_state[k] = v.detach().cpu().clone()

# delete model instance and clear GPU memory
del mdl_d
torch.cuda.empty_cache()

# load saved detector model onto GPU
print(f"[INFO] Reloading detector from '{SAVE_DIR}' onto GPU …")
reloaded = AutoModelForSequenceClassification.from_pretrained(
    SAVE_DIR,
    trust_remote_code=True,
    local_files_only=True,
    torch_dtype=torch.float32,
    device_map="auto",
    low_cpu_mem_usage=True
).eval()

# compute logits with reloaded model
with torch.no_grad():
    new_logits = reloaded(**enc).logits.detach().cpu()

# calculate maximum absolute difference in logits
logit_diff = (orig_logits - new_logits).abs().max().item()
print(f"\n[LOGITS] max |orig–reloaded| = {logit_diff:.3e}")

# compare each parameter between original and reloaded model
new_state = reloaded.state_dict()
param_diffs = {}
for k, orig_v in orig_state.items():
    if k not in new_state:
        print(f"[PARAM] Missing key in reloaded model: {k}")
        continue
    new_v = new_state[k].cpu()
    diff = (orig_v - new_v).abs().max().item()
    param_diffs[k] = diff

# summarize parameter differences
all_diffs = np.array(list(param_diffs.values()))
print(f"\n[PARAM] compared {len(param_diffs)} params")
print(f"[PARAM] max diff = {all_diffs.max():.3e}")
print(f"[PARAM] mean diff = {all_diffs.mean():.3e}")
print(f"[PARAM] top 5 largest diffs:")
for k, d in sorted(param_diffs.items(), key=lambda x: x[1], reverse=True)[:5]:
    print(f"   • {k}: {d:.3e}")

print("\n✅ verification complete.")