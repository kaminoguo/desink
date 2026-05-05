"""Appendix C: LoRA layer selection via raw vs de-sinked CKA on Pythia-1B."""

import torch
import numpy as np
import json
import os
import time
from sklearn.decomposition import PCA


def desink(X):
    """GPU-accelerated desink."""
    if isinstance(X, np.ndarray): X = torch.from_numpy(X).cuda()
    X_c = X - X.mean(dim=0, keepdim=True)
    _, _, Vt = torch.linalg.svd(X_c, full_matrices=False)
    d = Vt[0]; d = d / (d.norm() + 1e-10)
    return (X - (X @ d).unsqueeze(-1) * d.unsqueeze(0)).cpu().numpy()


def compute_linear_cka(X, Y):
    X = X - X.mean(axis=0, keepdims=True)
    Y = Y - Y.mean(axis=0, keepdims=True)
    hsic_xy = np.linalg.norm(X.T @ Y, 'fro') ** 2
    hsic_xx = np.linalg.norm(X.T @ X, 'fro') ** 2
    hsic_yy = np.linalg.norm(Y.T @ Y, 'fro') ** 2
    denom = np.sqrt(hsic_xx * hsic_yy)
    return float(hsic_xy / denom) if denom > 1e-10 else 0.0


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    MODEL = "EleutherAI/pythia-1b"  # 1B for LoRA fine-tuning feasibility
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Step 1: Extract hidden states for CKA
    print(f"\nLoading {MODEL} for CKA computation...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.float16, device_map="auto")
    model.eval()
    n_layers = model.config.num_hidden_layers

    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    texts = [t for t in ds["text"] if len(t.strip()) > 50][:200]

    layer_hidden = {}
    for idx, text in enumerate(texts):
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        with torch.no_grad():
            out = model(**inputs, output_hidden_states=True)
        for li in range(n_layers + 1):
            h = out.hidden_states[li][0].float().cpu().numpy()
            if li not in layer_hidden:
                layer_hidden[li] = []
            layer_hidden[li].append(h)
        if (idx + 1) % 50 == 0:
            print(f"  {idx+1}/{len(texts)}")

    for li in layer_hidden:
        layer_hidden[li] = np.concatenate(layer_hidden[li], axis=0)

    # Step 2: Compute per-layer CKA similarity with neighbors
    print("\nComputing CKA-based layer importance...")

    def layer_cka_scores(do_desink=False):
        """Higher CKA with neighbors = more redundant = skip for LoRA."""
        scores = {}
        for li in range(n_layers):
            X = layer_hidden[li]
            Y = layer_hidden[li + 1]
            if do_desink:
                X = desink(X)
                Y = desink(Y)
            scores[li] = compute_linear_cka(X, Y)
        return scores

    cka_raw = layer_cka_scores(do_desink=False)
    cka_ds = layer_cka_scores(do_desink=True)

    # Higher CKA = more redundant = SKIP for LoRA (don't put adapter there)
    # Lower CKA = more unique = PUT LoRA adapter here
    # Select top-K layers with LOWEST CKA for LoRA
    n_lora = n_layers // 4  # 25% of layers get LoRA

    rank_raw = sorted(cka_raw.keys(), key=lambda k: cka_raw[k])  # ascending CKA
    rank_ds = sorted(cka_ds.keys(), key=lambda k: cka_ds[k])

    lora_raw = set(rank_raw[:n_lora])  # lowest CKA = most unique
    lora_ds = set(rank_ds[:n_lora])

    print(f"\n{'Layer':<8} {'CKA_raw':>10} {'CKA_ds':>10}")
    print("-" * 30)
    for li in range(n_layers):
        marker = ""
        if li in lora_raw and li not in lora_ds:
            marker = " ← RAW-only LoRA"
        elif li not in lora_raw and li in lora_ds:
            marker = " ← DS-only LoRA"
        elif li in lora_raw and li in lora_ds:
            marker = " ← BOTH LoRA"
        print(f"L{li:<7} {cka_raw[li]:>9.4f} {cka_ds[li]:>9.4f}{marker}")

    print(f"\nLoRA layers (raw CKA):     {sorted(lora_raw)}")
    print(f"LoRA layers (de-sinked):   {sorted(lora_ds)}")
    overlap = len(lora_raw & lora_ds)
    print(f"Overlap: {overlap}/{n_lora}")

    # Step 3: Fine-tune with LoRA on each selection
    del model
    torch.cuda.empty_cache()

    from datasets import load_dataset
    train_ds = load_dataset("ag_news", split="train[:2000]")
    eval_ds_ag = load_dataset("ag_news", split="test[:500]")

    def finetune_lora(target_layers, label):
        print(f"\n--- LoRA fine-tuning: {label} (layers {sorted(target_layers)}) ---")
        from peft import LoraConfig, get_peft_model, TaskType
        from transformers import TrainingArguments, Trainer

        base_model = AutoModelForCausalLM.from_pretrained(
            MODEL, torch_dtype=torch.float16, device_map="auto")

        # Build target_modules for specific layers
        target_modules = []
        for li in target_layers:
            target_modules.extend([
                f"gpt_neox.layers.{li}.attention.query_key_value",
                f"gpt_neox.layers.{li}.mlp.dense_h_to_4h",
            ])

        config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=8,
            lora_alpha=16,
            target_modules=target_modules,
            lora_dropout=0.05,
        )
        peft_model = get_peft_model(base_model, config)
        trainable = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
        print(f"  Trainable params: {trainable:,}")

        # Tokenize
        def tokenize(examples):
            texts_combined = [f"{t} Label: {l}" for t, l in zip(examples["text"], examples["label"])]
            tok = tokenizer(texts_combined, truncation=True, max_length=128, padding="max_length")
            tok["labels"] = tok["input_ids"].copy()
            return tok

        train_tokenized = train_ds.map(tokenize, batched=True, remove_columns=train_ds.column_names)
        eval_tokenized = eval_ds_ag.map(tokenize, batched=True, remove_columns=eval_ds_ag.column_names)

        args = TrainingArguments(
            output_dir=f"logs/lora/{label}",
            num_train_epochs=1,
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            learning_rate=2e-4,
            logging_steps=50,
            save_strategy="no",
            fp16=False,
            report_to="none",
        )
        trainer = Trainer(model=peft_model, args=args,
                         train_dataset=train_tokenized, eval_dataset=eval_tokenized)
        trainer.train()
        metrics = trainer.evaluate()
        ppl = np.exp(metrics["eval_loss"])
        print(f"  Eval PPL: {ppl:.2f}")

        del peft_model, base_model
        torch.cuda.empty_cache()
        return ppl

    try:
        from peft import LoraConfig
        ppl_raw = finetune_lora(lora_raw, "raw_cka")
        ppl_ds = finetune_lora(lora_ds, "desink_cka")

        print(f"\n{'='*60}")
        print("LoRA LAYER SELECTION RESULTS")
        print(f"{'='*60}")
        print(f"Raw CKA LoRA PPL:     {ppl_raw:.2f} (layers {sorted(lora_raw)})")
        print(f"De-sinked CKA LoRA PPL: {ppl_ds:.2f} (layers {sorted(lora_ds)})")

        if ppl_ds < ppl_raw:
            print(f"\n→ DE-SINKED SELECTION IS BETTER by {ppl_raw - ppl_ds:.2f} PPL")
        else:
            print(f"\n→ Raw selection is better or equal")

    except ImportError:
        print("\npeft not installed, skipping LoRA fine-tuning. Install with: pip install peft")
        ppl_raw, ppl_ds = None, None

    os.makedirs("logs/lora_selection", exist_ok=True)
    out = {
        "model": MODEL, "n_layers": n_layers, "n_lora": n_lora,
        "cka_raw": {str(k): v for k, v in cka_raw.items()},
        "cka_desink": {str(k): v for k, v in cka_ds.items()},
        "lora_raw": sorted(lora_raw), "lora_ds": sorted(lora_ds),
        "ppl_raw": ppl_raw, "ppl_ds": ppl_ds,
    }
    with open("logs/lora_selection/lora_selection.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved.")


if __name__ == "__main__":
    main()
