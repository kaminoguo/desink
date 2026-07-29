"""Build the single shared tokenized corpus for the T1 rerun.

Source  : Skylion007/openwebtext via its pinned HF parquet conversion
          (refs/convert/parquet) — same content as the streaming loader used by
          the pythia-70m batch (exp_b4_v3), but deterministic and reproducible.
Protocol: documents tokenized and CONCATENATED (no separator), then chunked
          into fixed 512-token sequences.  Matches the 70M protocol.
Reuse   : Pythia 160m/410m/1.4b share one tokenizer -> one corpus for every
          (model, checkpoint).  Cross-checkpoint curves are only comparable if
          the corpus is bit-identical, so it is built once and memmapped.

Layout of corpus.npy  [N_SEQ, 512] int32:
    rows    0..391   seed 0   (200,704 tokens)
    rows  392..783   seed 1
    rows  784..1175  seed 2
    rows 1176..1183  loss probe (held out of all metrics)
    rows    0..999   used by the E3 token-count sweep (<=512K tokens)

This script FAILS LOUDLY.  The bug that voided the previous run was a bare
`except` swallowing a dataset-load error and substituting 200 copies of one
sentence; every guard below exists to make that class of failure impossible.
"""
import os, sys, json, hashlib, traceback

os.environ.setdefault("HF_HOME", "/workspace/desink/hf")
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer

SEQ_LEN = 512
N_SEQ_PER_SEED = 392            # 200,704 tokens per seed
N_SEEDS = 3                     # 602,112 tokens total ~= the 600K spec
N_PROBE_SEQ = 8                 # held-out slice for the loss sanity probe
N_SEQ = N_SEQ_PER_SEED * N_SEEDS + N_PROBE_SEQ
N_TOKENS = N_SEQ * SEQ_LEN

OUT_DIR = "/workspace/desink/corpus"
TOKENIZER = "EleutherAI/pythia-160m"


def load_documents(n_needed_tokens):
    """Return a list of raw OpenWebText document strings.  Raises on failure."""
    shards, docs, approx_tokens = [], [], 0
    for shard_idx in range(4):                       # 1 shard is already plenty
        fn = f"plain_text/train/{shard_idx:04d}.parquet"
        path = hf_hub_download("Skylion007/openwebtext", fn,
                               repo_type="dataset", revision="refs/convert/parquet")
        shards.append(fn)
        tbl = pq.read_table(path, columns=["text"])
        for t in tbl.column("text").to_pylist():
            docs.append(t)
            approx_tokens += len(t) // 4             # ~4 chars/token, rough
            if approx_tokens > n_needed_tokens * 2:  # 2x margin
                return docs, shards
    return docs, shards


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(TOKENIZER)

    print(f"[corpus] fetching OpenWebText parquet ...", flush=True)
    docs, shards = load_documents(N_TOKENS)
    print(f"[corpus] {len(docs)} documents from {shards}", flush=True)
    if len(docs) < 100:
        raise RuntimeError(f"only {len(docs)} documents — refusing to proceed")

    ids = []
    B = 256
    for i in range(0, len(docs), B):
        for seq in tok(docs[i:i + B])["input_ids"]:
            ids.extend(seq)
        if len(ids) >= N_TOKENS:
            break
    print(f"[corpus] tokenized {len(ids)} tokens (need {N_TOKENS})", flush=True)
    if len(ids) < N_TOKENS:
        raise RuntimeError(f"corpus too short: {len(ids)} < {N_TOKENS}")

    arr = np.asarray(ids[:N_TOKENS], dtype=np.int32).reshape(N_SEQ, SEQ_LEN)

    # ── guards against the 2026-Q1 "200 identical sentences" failure ──────────
    uniq_ids = np.unique(arr)
    n_dup_rows = N_SEQ - np.unique(arr, axis=0).shape[0]
    per_seq_uniq = np.array([len(np.unique(r)) / SEQ_LEN for r in arr])
    checks = {
        "n_unique_token_ids": int(uniq_ids.size),
        "n_duplicate_sequences": int(n_dup_rows),
        "per_seq_unique_frac_mean": float(per_seq_uniq.mean()),
        "per_seq_unique_frac_min": float(per_seq_uniq.min()),
        "vocab_coverage": float(uniq_ids.size / len(tok)),
    }
    print("[corpus] checks:", json.dumps(checks, indent=2), flush=True)
    assert checks["n_unique_token_ids"] >= 5000, "corpus vocabulary collapsed"
    assert checks["n_duplicate_sequences"] == 0, "duplicate 512-token sequences"
    assert checks["per_seq_unique_frac_mean"] >= 0.30, "sequences too repetitive"

    sha = hashlib.sha256(arr.tobytes()).hexdigest()[:16]
    np.save(f"{OUT_DIR}/corpus.npy", arr)
    meta = {
        "source": "Skylion007/openwebtext @ refs/convert/parquet",
        "shards": shards, "tokenizer": TOKENIZER,
        "seq_len": SEQ_LEN, "n_seq": N_SEQ, "n_tokens": N_TOKENS,
        "n_seq_per_seed": N_SEQ_PER_SEED, "n_seeds": N_SEEDS,
        "n_probe_seq": N_PROBE_SEQ,
        "packing": "documents concatenated with no separator, then chunked",
        "sha256_16": sha, "checks": checks,
    }
    with open(f"{OUT_DIR}/corpus_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    # ── eyeball gate: decoded samples must read as real English prose ────────
    for i in (0, 392, 784, 1176):
        print(f"\n[corpus] --- seq {i} ---\n{tok.decode(arr[i][:120])!r}", flush=True)
    print(f"\n[corpus] OK sha={sha}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
