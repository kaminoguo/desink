#!/bin/bash
# On-box watchdog (cron */2).
#
# Gate: nothing GPU-side is launched unless cuInit() actually succeeds.  On
# 2026-07-29 this machine's host-side nvidia_uvm module went into EIO after a
# GPU fell off the PCIe bus — nvidia-smi still listed 4 healthy cards while
# open("/dev/nvidia-uvm") returned errno 5 and cuInit returned 999, so a naive
# "is nvidia-smi ok" check would have spun jobs that all die in 6 s.  Probing
# the driver directly is the only honest liveness test.
#
# Once CUDA is back the run resumes by itself: weights and corpora are already
# on disk and every stage is idempotent on its result JSON.
R=/workspace/desink
export HF_TOKEN="${HF_TOKEN:?}"
export HF_HOME=/workspace/.hf_home
cd "$R" || exit 1

NREADY=$(ls "$R"/ready 2>/dev/null | wc -l)
NE1=$(ls "$R"/results/e1_*.json "$R"/results/e3_*.json 2>/dev/null | wc -l)
NE2=$(ls "$R"/results/e2_*.json 2>/dev/null | wc -l)
E1TOT=63; E2TOT=13
FREE=$(df --output=avail -BG / | tail -1 | tr -dc 0-9)

CUDA=$(timeout 90 "$R"/venv/bin/python - <<'EOF' 2>/dev/null
import ctypes
try:
    l = ctypes.CDLL("libcuda.so.1")
    r = l.cuInit(0)
    n = ctypes.c_int()
    l.cuDeviceGetCount(ctypes.byref(n))
    print("OK" if r == 0 and n.value > 0 else f"DEAD(cuInit={r})")
except Exception as e:
    print(f"DEAD({e})")
EOF
)
[ -z "$CUDA" ] && CUDA="DEAD(probe-timeout)"

if [ "$CUDA" = "OK" ]; then
  if [ "$NE1" -lt "$E1TOT" ]; then
    if ! pgrep -f "code/orchestrate.py" >/dev/null; then
      echo "$(date -Is) START E1 ($NE1/$E1TOT)" >> "$R"/logs/watchdog.log
      tmux kill-session -t orch 2>/dev/null
      tmux new-session -d -s orch "./venv/bin/python code/orchestrate.py >> logs/orchestrate.log 2>&1"
    fi
  elif [ "$NE2" -lt "$E2TOT" ] && ! pgrep -f "e2_orchestrate.py" >/dev/null; then
    echo "$(date -Is) START E2 ($NE2/$E2TOT)" >> "$R"/logs/watchdog.log
    ./venv/bin/python code/reap.py >> logs/reap.log 2>&1
    tmux kill-session -t orch2 2>/dev/null
    tmux new-session -d -s orch2 "cd $R/code && $R/venv/bin/python e2_orchestrate.py >> $R/logs/orchestrate_e2.log 2>&1"
  fi
  [ "$FREE" -lt 40 ] && ./venv/bin/python code/reap.py >> logs/reap.log 2>&1
else
  # GPUs unusable: make sure nothing is spinning and burning retries
  pkill -f "code/orchestrate.py" 2>/dev/null
  pkill -f "e2_orchestrate.py" 2>/dev/null
fi

# weights/corpora are prerequisites and need no GPU — keep them topped up
if ! pgrep -f "code/prefetch.py" >/dev/null && [ "$NREADY" -lt 60 ]; then
  tmux new-session -d -s prefetch "./venv/bin/python code/prefetch.py >> logs/prefetch.log 2>&1"
fi

GPU=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | tr '\n' ',')
echo "$(date -Is) cuda=$CUDA e1=$NE1/$E1TOT e2=$NE2/$E2TOT ready=$NREADY gpu=[$GPU] free=${FREE}G" \
  >> "$R"/logs/watchdog.log
