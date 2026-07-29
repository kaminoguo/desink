#!/bin/bash
# Bring a fresh vast box to the state the T1 rerun needs.  Idempotent.
# Uses the box's preinstalled torch via --system-site-packages so nothing large
# is downloaded and the numerics match the image the earlier results came from.
set -e
R=/workspace/desink
mkdir -p $R/{code,corpus,results,logs/jobs,ready,ready_e2}
[ -d $R/venv ] || uv venv $R/venv --python /venv/main/bin/python --system-site-packages
VIRTUAL_ENV=$R/venv uv pip install -q pyarrow
$R/venv/bin/python - <<'EOF'
import torch, transformers, pyarrow, sys
print("python", sys.version.split()[0], "torch", torch.__version__,
      "cuda", torch.version.cuda, "gpus", torch.cuda.device_count(),
      "transformers", transformers.__version__, "pyarrow", pyarrow.__version__)
assert torch.cuda.device_count() >= 1
for i in range(torch.cuda.device_count()):
    a = torch.randn(1024, 1024, device=f"cuda:{i}")
    print(f"  gpu{i} {torch.cuda.get_device_name(i)} matmul_ok={bool((a@a).isfinite().all())}")
EOF
echo "SETUP OK"
