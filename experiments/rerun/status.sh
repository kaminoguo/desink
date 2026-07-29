#!/bin/bash
# One-shot progress snapshot.  Kept short so a flaky ssh link can always fetch it.
R=/workspace/desink
cd "$R" || exit 1
echo "== $(date -Is)  results=$(ls results/e*.json 2>/dev/null|wc -l)/63  ready=$(ls ready|wc -l)/60  disk=$(df -h /|awk 'NR==2{print $4}')"
echo "== gpu: $(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader|tr '\n' '|')"
echo "== running:"
for f in $(ls -t logs/jobs/*.log 2>/dev/null | head -6); do
  printf '%-34s %s\n' "$(basename "$f" .log)" "$(grep -hv Warning "$f" | grep -E 'seed|n[0-9]+ |DONE|Error|Traceback' | tail -1)"
done
echo "== finished (last 8):"
grep -hE '^\[gpu' logs/orchestrate.log 2>/dev/null | tail -8
echo "== failures:"
grep -lE 'Traceback|Error' logs/jobs/*.log 2>/dev/null | head -5
