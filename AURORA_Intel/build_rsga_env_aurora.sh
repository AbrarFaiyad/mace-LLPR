#!/bin/bash
# Build the Aurora MACE-RSGA training venv. Run ONCE on a login node.
#
#   bash build_rsga_env_aurora.sh
#
# (For the LLPR pipeline venv used by Auto-Finetuner, use
#  build_env_llpr_aurora.sh instead — that builds the mace-LLPR fork.)
#
# Creates the RSGA training venv on top of the Aurora py-torch/2.10.0 module
# (XPU-enabled). Installs the RSGA fork of mace via --no-deps to avoid
# clobbering the Aurora-provided torch.
#
# After install we patch site-packages/mace/tools/distributed_tools.py so
# that (a) the xpu code path uses the xccl backend (Aurora replaced
# ccl with xccl), and (b) the mpi launcher also recognises Intel
# MPI/PALS env vars (PMI_RANK, PALS_RANKID, PALS_LOCAL_SIZE).
#
# Paths are configurable via env vars; edit the defaults or export them:
#   VENV_DIR      where the venv is created  (default: $HOME/venv_mace_rsga)
#   RSGA_SRC_DIR  RSGA fork checkout         (default: $HOME/RSGA_MACE_OPT)
#   RSGA_REPO     RSGA fork git URL

set -e

VENV_DIR="${VENV_DIR:-$HOME/venv_mace_rsga}"
RSGA_SRC_DIR="${RSGA_SRC_DIR:-$HOME/RSGA_MACE_OPT}"
RSGA_REPO="${RSGA_REPO:-https://github.com/GSLab2025/RSGA_MACE_OPT.git}"

echo "==> loading Aurora modules"
module load py-torch/2.10.0
module load py-numpy/2.3.4
module load py-scipy/1.16.3
module load py-matplotlib/3.10.7
module load py-pyyaml/6.0.3

if [ ! -d "${VENV_DIR}" ]; then
    echo "==> creating venv at ${VENV_DIR} (with --system-site-packages)"
    python -m venv --system-site-packages "${VENV_DIR}"
else
    echo "==> venv already exists at ${VENV_DIR}, reusing"
fi
source "${VENV_DIR}/bin/activate"

echo "==> base toolchain checks"
python -c "import torch, sys; print('python', sys.version.split()[0]); print('torch', torch.__version__); print('xpu_compiled', getattr(torch.version, 'xpu', 'no'))"

echo "==> pip-installing leaf deps (system torch/numpy/scipy/pyyaml inherited)"
pip install --no-cache-dir \
    ase torchmetrics torch-ema matscipy opt-einsum-fx \
    prettytable python-hostlist configargparse h5py tqdm lmdb orjson pandas

echo "==> cloning / updating RSGA fork at ${RSGA_SRC_DIR}"
if [ ! -d "${RSGA_SRC_DIR}/.git" ]; then
    git clone "${RSGA_REPO}" "${RSGA_SRC_DIR}"
else
    (cd "${RSGA_SRC_DIR}" && git fetch && git pull)
fi

echo "==> pip-installing RSGA --no-deps (editable)"
pip install --no-deps -e "${RSGA_SRC_DIR}/mace"

echo "==> pip-installing e3nn 0.4.4 (RSGA pinned) --no-deps"
pip install --no-deps "e3nn==0.4.4"

echo "==> applying Aurora patches (xccl + Intel MPI env vars)"
PATCH_SCRIPT="$(cd "$(dirname "$0")" && pwd)/patch_distributed_for_aurora.py"
if [ ! -f "${PATCH_SCRIPT}" ]; then
    echo "ERROR: missing ${PATCH_SCRIPT}"
    exit 1
fi
python "${PATCH_SCRIPT}" "${RSGA_SRC_DIR}/mace"

echo "==> sanity import"
python -c "
import mace, inspect, torch
print('mace:', mace.__version__, mace.__file__)
print('torch:', torch.__version__, 'has_xpu_attr:', hasattr(torch, 'xpu'))
from mace.tools import distributed_tools as dt
src = inspect.getsource(dt.init_distributed)
assert 'xccl' in src, 'patch missed: xccl backend not present'
assert 'PALS_RANKID' in src or 'PMI_RANK' in src, 'patch missed: Intel MPI env vars'
print('patch verified: xccl + Intel MPI env vars present')

from mace.cli import run_train as rt
src2 = inspect.getsource(rt.run)
assert 'if not hasattr(torch, \"xpu\")' in src2, 'patch missed: ipex optional'
print('patch verified: ipex/oneccl imports made optional')
"

echo "Done. To use this venv:"
echo "  source $(cd "$(dirname "$0")" && pwd)/env_aurora.sh"
