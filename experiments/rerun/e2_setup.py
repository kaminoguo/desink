"""E2 (OLMo-2-0425-1B-early-training) setup: OLMo-tokenised corpus + weight prefetch.

Same documents, same packing, same 1184x512 layout as the Pythia corpus — only
the tokenizer differs (OLMo vocab 100352 vs Pythia 50304), so the two corpora
cover slightly different amounts of text.  That is unavoidable across tokenizer
families; the protocol held fixed is "same source documents, same packing,
same sequence length, same token budget".

Usage:  e2_setup.py corpus     # build /workspace/desink/corpus/corpus_olmo.npy
        e2_setup.py prefetch   # download the 13 revisions
"""
import os, sys, json, time, hashlib, traceback

os.environ.setdefault("HF_HOME", "/workspace/.hf_home")

REPO = "allenai/OLMo-2-0425-1B-early-training"
# 13 points spread over the available stage-1 grid (step0 .. step36000)
REVS = ["stage1-step0-tokens0B", "stage1-step1000-tokens3B", "stage1-step2000-tokens5B",
        "stage1-step3000-tokens7B", "stage1-step4000-tokens9B", "stage1-step6000-tokens13B",
        "stage1-step8000-tokens17B", "stage1-step11000-tokens24B", "stage1-step15000-tokens32B",
        "stage1-step20000-tokens42B", "stage1-step26000-tokens55B", "stage1-step31000-tokens66B",
        "stage1-step36000-tokens76B"]

SEQ_LEN, N_SEQ = 512, 392 * 3 + 8
N_TOKENS = SEQ_LEN * N_SEQ
READY = "/workspace/desink/ready_e2"
OUT = "/workspace/desink/corpus/corpus_olmo.npy"


def build_corpus():
    import numpy as np, pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(REPO, revision=REVS[1])
    path = hf_hub_download("Skylion007/openwebtext", "plain_text/train/0000.parquet",
                           repo_type="dataset", revision="refs/convert/parquet")
    docs = pq.read_table(path, columns=["text"]).column("text").to_pylist()
    print(f"[e2] {len(docs)} docs, vocab={len(tok)}", flush=True)

    ids = []
    for i in range(0, len(docs), 256):
        for seq in tok(docs[i:i + 256])["input_ids"]:
            ids.extend(seq)
        if len(ids) >= N_TOKENS:
            break
    if len(ids) < N_TOKENS:
        raise RuntimeError(f"corpus too short: {len(ids)} < {N_TOKENS}")

    arr = np.asarray(ids[:N_TOKENS], dtype=np.int32).reshape(N_SEQ, SEQ_LEN)
    uniq = int(np.unique(arr).size)
    dup = N_SEQ - int(np.unique(arr, axis=0).shape[0])
    frac = float(np.mean([len(np.unique(r)) / SEQ_LEN for r in arr]))
    print(f"[e2] unique_ids={uniq} dup_seq={dup} uniq_frac={frac:.3f}", flush=True)
    assert uniq >= 5000 and dup == 0 and frac >= 0.30, "corpus validity check failed"

    np.save(OUT, arr)
    json.dump({"source": "Skylion007/openwebtext @ refs/convert/parquet shard 0000",
               "tokenizer": REPO, "seq_len": SEQ_LEN, "n_seq": N_SEQ,
               "n_seq_per_seed": 392, "n_seeds": 3, "n_probe_seq": 8,
               "n_tokens": N_TOKENS,
               "packing": "documents concatenated with no separator, then chunked",
               "sha256_16": hashlib.sha256(arr.tobytes()).hexdigest()[:16],
               "checks": {"n_unique_token_ids": uniq, "n_duplicate_sequences": dup,
                          "per_seq_unique_frac_mean": frac}},
              open(OUT.replace(".npy", "_meta.json"), "w"), indent=2)
    print(f"[e2] corpus OK -> {OUT}", flush=True)
    print(repr(tok.decode(arr[0][:80])), flush=True)


def prefetch():
    from concurrent.futures import ThreadPoolExecutor
    from huggingface_hub import snapshot_download
    os.makedirs(READY, exist_ok=True)
    allow = ["*.safetensors", "*.safetensors.index.json", "config.json",
             "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"]

    def one(rev):
        mk = f"{READY}/{rev}.done"
        if os.path.exists(mk):
            return
        for a in range(4):
            try:
                t = time.time()
                snapshot_download(REPO, revision=rev, allow_patterns=allow, max_workers=8)
                open(mk, "w").write("ok")
                print(f"[e2 dl] {rev} {time.time()-t:.0f}s", flush=True)
                return
            except Exception:
                traceback.print_exc()
                time.sleep(10 * (a + 1))
        print(f"[e2 dl] GIVING UP {rev}", flush=True)

    with ThreadPoolExecutor(max_workers=3) as ex:
        list(ex.map(one, REVS))
    print("[e2 dl] ALL DONE", flush=True)


if __name__ == "__main__":
    {"corpus": build_corpus, "prefetch": prefetch}[sys.argv[1]]()
