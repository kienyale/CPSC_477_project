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

# enable debug logging of device and cuda memory
DEBUG = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[debug] using device: {device}, torch {torch.__version__}")

# helper to log current GPU memory usage
def log_memory_info(ctx=""):
    if DEBUG and device.type == "cuda":
        allocated = torch.cuda.memory_allocated(device) / (1024**2)
        reserved  = torch.cuda.memory_reserved(device) / (1024**2)
        print(f"[mem] {ctx}: alloc {allocated:.1f}MB, resv {reserved:.1f}MB")

# initialize embedder for semantic similarity calculations
embedder = SentenceTransformer("all-MiniLM-L6-v2", device=device)

def semantic_similarity(a, b):
    # compute normalized cosine similarity between two text embeddings
    e1 = embedder.encode(a, convert_to_tensor=True, show_progress_bar=False)
    e2 = embedder.encode(b, convert_to_tensor=True, show_progress_bar=False)
    return ((F.cosine_similarity(e1, e2, dim=0) + 1) / 2).item()

def compute_length_penalty(txt, ideal_len):
    # penalize generations exceeding the target word count
    return 0.01 * max(0, len(txt.split()) - ideal_len)

def top_p_filtering(logits, top_p=0.9, filter_value=-1e4):
    # perform nucleus (top-p) filtering on logits
    sorted_logits, idx = torch.sort(logits, descending=True)
    probs = torch.softmax(sorted_logits, dim=-1)
    cum_probs = torch.cumsum(probs, dim=-1)
    mask = cum_probs > top_p
    mask[..., 1:] = mask[..., :-1].clone()
    mask[..., 0] = False
    sorted_logits[mask] = filter_value
    return torch.empty_like(logits).scatter(-1, idx, sorted_logits)

def sample_sequence_with_log_probs(model, tok, input_ids, max_len=8, top_p=0.9, window=128):
    # generate sequence with cumulative log probability
    model.eval()
    context = input_ids if input_ids.shape[-1] <= window else input_ids[:, -window:]
    generated, log_probs = context, []
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
        log_probs.append(lps.gather(1, next_token).squeeze(1))
        generated = torch.cat([generated, next_token], dim=-1)
        if generated.shape[-1] > window:
            generated = generated[:, -window:]
        if next_token.item() == tok.eos_token_id:
            break
    if not log_probs:
        return generated, torch.zeros(1, device=device)
    return generated, torch.stack(log_probs, dim=1).sum(dim=1)

# load and prepare dataset for training
df = pd.read_csv(
    "train_baseline.csv",
    names=["problem", "type", "level", "ai_solution", "human_solution"],
    header=0
)
ai_sols = df["ai_solution"].tolist()
hu_sols = df["human_solution"].tolist()
print(f"[debug] loaded {len(ai_sols)} training examples")

# configure humanizer model for supervised fine-tuning
cfg_h = AutoConfig.from_pretrained("Qwen/Qwen2.5-1.5B", trust_remote_code=True)
cfg_h.use_cache = False

tok_h = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B", trust_remote_code=True)
mdl_h = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-1.5B",
    config=cfg_h,
    trust_remote_code=True,
    torch_dtype=torch.float16,
    device_map=None,
    low_cpu_mem_usage=False
).to(device)

# apply LoRA adapters to humanizer
lora_cfg = LoraConfig(r=8, lora_alpha=32, target_modules=["q_proj", "k_proj", "v_proj"], lora_dropout=0.1, bias="none")
mdl_h = get_peft_model(mdl_h, lora_cfg)

def prep_h(batch):
    # tokenize AI prompts and human references
    prompts = ["Rewrite this solution to make it more human: " + a for a in batch["ai_solution"]]
    enc = tok_h(prompts, truncation=True, padding="longest", max_length=128)
    lbl = tok_h(batch["human_solution"], truncation=True, padding="longest", max_length=128)
    enc["labels"] = lbl["input_ids"]
    return enc

hf_train = Dataset.from_pandas(df)
sft_ds_h = hf_train.map(prep_h, batched=True, remove_columns=df.columns.tolist())
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
tr_h = Trainer(model=mdl_h, args=args_h, train_dataset=sft_ds_h, tokenizer=tok_h, data_collator=coll_h)
print("[debug] starting humanizer fine-tuning…")
tr_h.train()
tr_h.save_model("humanizer_sft")
print("[debug] humanizer fine-tuning complete.")

# configure detector model for supervised fine-tuning
cfg_d = AutoConfig.from_pretrained("Qwen/Qwen2.5-1.5B", trust_remote_code=True)
cfg_d.num_labels = 2
cfg_d.use_cache = False

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

# freeze base and train classification head only
for p in mdl_d.parameters(): p.requires_grad = False
for p in mdl_d.score.parameters(): p.requires_grad = True

# prepare detector training dataset
det_texts = ai_sols + hu_sols
det_labels = [0]*len(ai_sols) + [1]*len(hu_sols)
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
tr_d = Trainer(model=mdl_d, args=args_d, train_dataset=det_ds, tokenizer=tok_d, data_collator=coll_d)
print("[debug] starting detector fine-tuning…")
tr_d.train()
tr_d.save_model("detector_sft")
print("[debug] detector fine-tuning complete.")

# set up optimizers for adversarial rl loop
optimizer_h = AdamW(mdl_h.parameters(), lr=1e-5)
optimizer_d = AdamW(mdl_d.score.parameters(), lr=1e-5)

num_iters, batch_size, gen_max_len, window_size = 20, 8, 16, 128
task_prefix = "Rewrite this solution to make it more human: "
for itr in range(num_iters):
    print(f"\n===== rl iter {itr+1}/{num_iters} =====")
    mdl_h.train(); mdl_d.eval()
    rewards, pg_losses, adv_texts, adv_labels = [], [], [], []
    idxs = np.random.choice(len(ai_sols), batch_size, replace=False)
    for i in idxs:
        ai_txt, hu_txt = ai_sols[i], hu_sols[i]
        # generate human-like sample and log-prob
        enc_h = tok_h(task_prefix + ai_txt, return_tensors="pt", truncation=True, padding="longest", max_length=128).to(device)
        gen_ids, logp = sample_sequence_with_log_probs(mdl_h, tok_h, enc_h.input_ids, max_len=gen_max_len, top_p=0.9, window=window_size)
        gen_txt = tok_h.decode(gen_ids[0], skip_special_tokens=True)
        # evaluate with detector
        det_in = tok_d(gen_txt, return_tensors="pt", truncation=True, padding="longest", max_length=256).to(device)
        with torch.no_grad(): logits_d = mdl_d(**det_in).logits
        p_human = F.softmax(logits_d, dim=-1)[0,1].item()
        # compute reward and update humanizer
        sim = semantic_similarity(gen_txt, hu_txt)
        pen = compute_length_penalty(gen_txt, len(hu_txt.split()))
        raw_R = p_human - (1 - sim) - pen; R = float(np.clip(raw_R, -1.0, 1.0))
        rewards.append(R); loss_pg = -(logp * R).mean()
        optimizer_h.zero_grad(); loss_pg.backward(); optimizer_h.step(); pg_losses.append(loss_pg.item())
        # collect adversarial examples
        if np.random.rand() < 0.5:
            adv_texts += [hu_txt, gen_txt]; adv_labels += [1, 0]
        else:
            adv_texts += [gen_txt, hu_txt]; adv_labels += [0, 1]
    print(f" humanizer ▶ avg_reward={np.mean(rewards):.4f}, avg_pg_loss={np.mean(pg_losses):.4f}")
    # train detector on adversarial batch
    mdl_d.train()
    batch = tok_d(adv_texts, truncation=True, padding="longest", max_length=256, return_tensors="pt").to(device)
    labels_t = torch.tensor(adv_labels, device=device)
    loss_d = F.cross_entropy(mdl_d(**batch).logits, labels_t)
    optimizer_d.zero_grad(); loss_d.backward(); optimizer_d.step()
    print(f" detector ▶ adv_loss={loss_d.item():.4f}")
print("\n✅ adversarial reinforcement learning complete.")

# verify detector persistence and reload
SAVE_DIR = "saved_detector_rl"
os.makedirs(SAVE_DIR, exist_ok=True)
print(f"[info] saving detector to '{SAVE_DIR}' …")
mdl_d.save_pretrained(SAVE_DIR); tok_d.save_pretrained(SAVE_DIR)

sample_texts = test_df["ai_solution"].tolist()[:4] + test_df["human_solution"].tolist()[:4]
enc = tok_d(sample_texts, return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)
mdl_d.eval()
with torch.no_grad(): orig_logits = mdl_d(**enc).logits.cpu()

# capture original parameters
from collections import OrderedDict
orig_state = OrderedDict({k: v.cpu().clone() for k, v in mdl_d.state_dict().items() if v.device.type != "meta"})
del mdl_d; torch.cuda.empty_cache()

print(f"[info] reloading detector from '{SAVE_DIR}' …")
reloaded = AutoModelForSequenceClassification.from_pretrained(SAVE_DIR, trust_remote_code=True, local_files_only=True, torch_dtype=torch.float32, device_map="auto", low_cpu_mem_usage=True).eval()
with torch.no_grad(): new_logits = reloaded(**enc).logits.cpu()

# compare logits and parameters
diff = (orig_logits - new_logits).abs().max().item()
print(f"\n[logits] max |orig–reloaded| = {diff:.3e}")
new_state = reloaded.state_dict()
param_diffs = {k: (orig_state[k] - new_state[k].cpu()).abs().max().item() for k in orig_state if k in new_state}
all_diffs = np.array(list(param_diffs.values()))
print(f"\n[param] compared {len(param_diffs)} params")
print(f"[param] max diff = {all_diffs.max():.3e}")
print(f"[param] mean diff = {all_diffs.mean():.3e}")
print("[param] top 5 diffs:")
for k, d in sorted(param_diffs.items(), key=lambda x: x[1], reverse=True)[:5]: print(f" • {k}: {d:.3e}")
print("\n✅ verification complete.")
