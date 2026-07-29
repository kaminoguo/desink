"""Appendix C: CKA-based distillation layer matching (GPT-2 to 6-layer student)."""

import torch
import numpy as np
import json
import os
import time


def desink(X):
    """GPU-accelerated desink."""
    if isinstance(X, np.ndarray): X = torch.from_numpy(X).cuda()
    X_c = X - X.mean(dim=0, keepdim=True)
    _, _, Vt = torch.linalg.svd(X_c, full_matrices=False)
    d = Vt[0]; d = d / (d.norm() + 1e-10)
    return (X - (X @ d).unsqueeze(-1) * d.unsqueeze(0)).cpu().numpy()


def linear_cka(X, Y):
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

    from transformers import AutoModelForCausalLM, AutoTokenizer, GPT2Config, GPT2LMHeadModel
    from datasets import load_dataset

    TEACHER = "gpt2"
    teacher_tokenizer = AutoTokenizer.from_pretrained(TEACHER)
    if teacher_tokenizer.pad_token is None:
        teacher_tokenizer.pad_token = teacher_tokenizer.eos_token

    teacher = AutoModelForCausalLM.from_pretrained(TEACHER, torch_dtype=torch.float32)
    teacher = teacher.to(device).eval()
    n_teacher = teacher.config.n_layer  # 12
    print(f"Teacher: {TEACHER}, {n_teacher} layers")

    # Create student (6-layer GPT-2)
    student_config = GPT2Config(
        n_layer=6, n_head=12, n_embd=768, vocab_size=teacher.config.vocab_size)
    n_student = student_config.n_layer

    # Load data
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    texts = [t for t in ds["text"] if len(t.strip()) > 50][:200]

    # Extract teacher hidden states for CKA
    print("\nExtracting teacher hidden states...")
    teacher_hidden = {}
    for idx, text in enumerate(texts):
        inputs = teacher_tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            out = teacher(**inputs, output_hidden_states=True)
        for li in range(n_teacher + 1):
            h = out.hidden_states[li][0].float().cpu().numpy()
            if li not in teacher_hidden:
                teacher_hidden[li] = []
            teacher_hidden[li].append(h)

    for li in teacher_hidden:
        teacher_hidden[li] = np.concatenate(teacher_hidden[li], axis=0)

    # Compute teacher layer-layer CKA to find best matching for 6 student layers
    print("\nComputing teacher CKA for layer matching...")

    def find_best_matching(do_desink=False):
        """Find best 6 teacher layers to match 6 student layers."""
        # For each student layer (0-5), find teacher layer with max CKA to
        # evenly-spaced teacher layers [0, 2, 4, 6, 8, 10]
        even_spacing = [int(i * n_teacher / n_student) for i in range(n_student)]

        # Compute CKA between all teacher layer pairs
        cka_matrix = np.zeros((n_teacher + 1, n_teacher + 1))
        for i in range(n_teacher + 1):
            for j in range(i, n_teacher + 1):
                X = teacher_hidden[i]
                Y = teacher_hidden[j]
                if do_desink:
                    X = desink(X)
                    Y = desink(Y)
                c = linear_cka(X, Y)
                cka_matrix[i, j] = c
                cka_matrix[j, i] = c

        # For each student layer position, find teacher layer with highest CKA
        # to the "expected" teacher layer at that relative depth
        matching = []
        for si in range(n_student):
            expected_teacher_layer = even_spacing[si]
            # Find teacher layer with highest CKA to this expected position
            best_tl = expected_teacher_layer
            best_cka = -1
            for tl in range(n_teacher + 1):
                if tl not in matching:  # don't reuse
                    c = cka_matrix[expected_teacher_layer, tl]
                    if c > best_cka:
                        best_cka = c
                        best_tl = tl
            matching.append(best_tl)
        return matching

    match_raw = find_best_matching(do_desink=False)
    match_ds = find_best_matching(do_desink=True)
    match_even = [int(i * n_teacher / n_student) for i in range(n_student)]

    print(f"Even spacing:    {match_even}")
    print(f"Raw CKA match:   {match_raw}")
    print(f"De-sinked match: {match_ds}")

    # Distill with each matching
    train_ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    train_texts = [t for t in train_ds["text"] if len(t.strip()) > 50][:2000]

    eval_ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")
    eval_text = "\n\n".join([t for t in eval_ds["text"] if len(t.strip()) > 0])

    def distill_with_matching(matching, label):
        print(f"\n--- Distilling with {label}: {matching} ---")
        student = GPT2LMHeadModel(student_config).to(device)

        # Initialize student layers from matched teacher layers
        teacher_layers = list(teacher.transformer.h)
        student_layers = list(student.transformer.h)
        for si, ti in enumerate(matching):
            if ti < len(teacher_layers):
                student_layers[si].load_state_dict(teacher_layers[ti].state_dict())
        # Copy embeddings
        student.transformer.wte.load_state_dict(teacher.transformer.wte.state_dict())
        student.transformer.wpe.load_state_dict(teacher.transformer.wpe.state_dict())
        student.lm_head.load_state_dict(teacher.lm_head.state_dict())

        # Fine-tune student
        optimizer = torch.optim.AdamW(student.parameters(), lr=5e-5)
        student.train()

        for epoch in range(2):
            total_loss = 0
            n_batches = 0
            for i in range(0, min(len(train_texts), 500), 1):
                text = train_texts[i]
                inputs = teacher_tokenizer(text, return_tensors="pt",
                                          truncation=True, max_length=128)
                inputs = {k: v.to(device) for k, v in inputs.items()}
                inputs["labels"] = inputs["input_ids"].clone()

                out = student(**inputs)
                loss = out.loss
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                total_loss += loss.item()
                n_batches += 1
            print(f"  Epoch {epoch}: loss={total_loss/n_batches:.4f}")

        # Evaluate PPL
        student.eval()
        eval_enc = teacher_tokenizer(eval_text, return_tensors="pt")
        input_ids = eval_enc.input_ids[:, :2048].to(device)
        nlls = []
        for i in range(0, input_ids.size(1) - 1, 512):
            end = min(i + 512, input_ids.size(1) - 1)
            ids = input_ids[:, i:end + 1]
            target = ids.clone()
            target[:, :-1] = -100
            with torch.no_grad():
                out = student(ids, labels=target)
                nlls.append(out.loss.item())
        ppl = np.exp(np.mean(nlls))
        print(f"  PPL: {ppl:.2f}")

        del student
        torch.cuda.empty_cache()
        return ppl

    ppl_even = distill_with_matching(match_even, "even_spacing")
    ppl_raw = distill_with_matching(match_raw, "raw_cka")
    ppl_ds = distill_with_matching(match_ds, "desink_cka")

    print(f"\n{'='*60}")
    print("CKA DISTILLATION RESULTS")
    print(f"{'='*60}")
    print(f"Even spacing PPL:     {ppl_even:.2f} (layers {match_even})")
    print(f"Raw CKA matching PPL: {ppl_raw:.2f} (layers {match_raw})")
    print(f"De-sinked CKA PPL:    {ppl_ds:.2f} (layers {match_ds})")

    if ppl_ds < ppl_raw and ppl_ds < ppl_even:
        print(f"\n→ DE-SINKED MATCHING IS BEST")
    elif ppl_raw < ppl_ds:
        print(f"\n→ Raw matching is better (matching doesn't need de-sink)")
    else:
        print(f"\n→ Even spacing is best (matching doesn't matter)")

    os.makedirs("logs/distillation", exist_ok=True)
    out = {
        "teacher": TEACHER, "n_teacher": n_teacher, "n_student": n_student,
        "match_even": match_even, "match_raw": match_raw, "match_ds": match_ds,
        "ppl_even": ppl_even, "ppl_raw": ppl_raw, "ppl_ds": ppl_ds,
    }
    with open("logs/distillation/distillation.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved.")


if __name__ == "__main__":
    main()
