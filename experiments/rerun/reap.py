"""Delete cached weights for checkpoints whose result JSON already exists.

Short-lived box + 158 GB of Pythia + 77 GB of OLMo does not fit alongside the
other project on this disk.  Reaping is safe because the orchestrators are
idempotent on results, not on weights: a reaped checkpoint with a result is
never re-run, and one without a result is never reaped.
"""
import os, sys, json, glob
os.environ.setdefault("HF_HOME", "/workspace/.hf_home")
from huggingface_hub import scan_cache_dir

RES = "/workspace/desink/results"
MIN_BYTES = 1024


def main(dry=False):
    have = set()
    for p in glob.glob(f"{RES}/e1_*.json") + glob.glob(f"{RES}/e2_*.json"):
        if os.path.getsize(p) < MIN_BYTES:
            continue
        d = json.load(open(p))
        have.add((d["model"], d["revision"]))
    # E3 reuses three 1.4b checkpoints that E1 also covers: pin them until the
    # E3 result exists, otherwise reaping forces a 17 GB re-download.
    needed = {("EleutherAI/pythia-1.4b", f"step{s}") for s in (512, 8000, 143000)
              if not os.path.exists(f"{RES}/e3_pythia-1.4b_step{s}.json")}

    cache = scan_cache_dir()
    kill = []
    for repo in cache.repos:
        if repo.repo_type != "model":
            continue
        for rev in repo.revisions:
            for ref in rev.refs:
                if (repo.repo_id, ref) in have and (repo.repo_id, ref) not in needed:
                    kill.append(rev.commit_hash)
    if not kill:
        print("[reap] nothing to reap")
        return
    strat = cache.delete_revisions(*kill)
    print(f"[reap] {len(kill)} revisions, freeing {strat.expected_freed_size_str}")
    if not dry:
        strat.execute()
        print("[reap] done")


if __name__ == "__main__":
    main(dry="--dry" in sys.argv)
