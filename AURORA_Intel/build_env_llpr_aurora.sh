#!/bin/bash
# Build the Aurora MACE-LLPR pipeline venv. Run ONCE on a login node.
#
#   bash build_env_llpr_aurora.sh
#
# Creates venv_mace_lora_llpr on top of the Aurora frameworks/2025.3.1 module
# (Python 3.12 + XPU torch 2.10 + native xccl). Installs our mace-LLPR fork
# (https://github.com/AbrarFaiyad/mace-LLPR) editable + --no-deps so the
# Aurora-provided torch/numpy/scipy are never clobbered, then applies the
# Aurora XPU/PALS patches and the SOAP+FPS deps the frame assembler needs.
#
# This is the venv the Auto-Finetuner active-learning pipeline uses
# (env_aurora.sh points VENV_PYTHON at it). For the separate RSGA training
# venv, use build_rsga_env_aurora.sh instead.
#
# Aurora adaptations applied (see AURORA_Intel/ in the fork for the reference
# copies + README):
#   1. distributed backend  ccl -> xccl   (Aurora renamed the Intel backend)
#   2. mpi launcher accepts PMI_RANK / PALS_RANKID  (Aurora launches via PALS)
#   3. ipex/oneccl imports made optional  (torch.xpu is native in torch 2.10)
#   4. xpu.set_device / DDP device_ids guarded for 1-tile-per-rank
#      (ZE_AFFINITY_MASK exposes a single tile, so device 0 is the only choice)

set -e

VENV_DIR="/home/afaiyad/QuantumDS/afaiyad/venv_mace_lora_llpr"
LLPR_SRC_DIR="/lus/flare/projects/QuantumDS/afaiyad/mace-LLPR-fork"
LLPR_REPO="git@github.com:AbrarFaiyad/mace-LLPR.git"
LLPR_BRANCH="main"

echo "==> loading Aurora framework module"
source /usr/share/lmod/lmod/init/bash 2>/dev/null || true
module load frameworks/2025.3.1

if [ ! -d "${VENV_DIR}" ]; then
    echo "==> creating venv at ${VENV_DIR} (with --system-site-packages)"
    python -m venv --system-site-packages "${VENV_DIR}"
else
    echo "==> venv already exists at ${VENV_DIR}, reusing"
fi
source "${VENV_DIR}/bin/activate"

echo "==> base toolchain checks"
python -c "import torch, sys; print('python', sys.version.split()[0]); print('torch', torch.__version__); print('has_xpu', hasattr(torch, 'xpu'))"

echo "==> pip-installing leaf deps (system torch/numpy/scipy/pyyaml inherited)"
# ase..pandas = MACE runtime deps;  dscribe/skmatter = SOAP+FPS for the
# frame_assembler (RC-binned candidate selection).
pip install --no-cache-dir \
    ase torchmetrics torch-ema matscipy opt-einsum-fx \
    prettytable python-hostlist configargparse h5py tqdm lmdb orjson pandas \
    "dscribe==2.1.2" "skmatter==0.3.3"

echo "==> cloning / updating mace-LLPR fork at ${LLPR_SRC_DIR}"
if [ ! -d "${LLPR_SRC_DIR}/.git" ]; then
    git clone "${LLPR_REPO}" "${LLPR_SRC_DIR}"
fi
(cd "${LLPR_SRC_DIR}" && git fetch && git checkout "${LLPR_BRANCH}" && git pull)

echo "==> pip-installing mace-LLPR --no-deps (editable)"
pip install --no-deps -e "${LLPR_SRC_DIR}"

echo "==> pip-installing e3nn 0.4.4 (pinned) --no-deps"
pip install --no-deps "e3nn==0.4.4"

echo "==> applying Aurora patches (xccl + PALS launcher + ipex-optional + DDP guard)"
# Prefer the copy kept inside the fork (AURORA_Intel/), fall back to the
# Auto-Finetuner repo copy so this builds even from a bare fork checkout.
PATCH_SCRIPT="${LLPR_SRC_DIR}/AURORA_Intel/patch_distributed_for_aurora.py"
if [ ! -f "${PATCH_SCRIPT}" ]; then
    PATCH_SCRIPT="$(cd "$(dirname "$0")" && pwd)/patch_distributed_for_aurora.py"
fi
if [ ! -f "${PATCH_SCRIPT}" ]; then
    echo "ERROR: patch script not found in fork AURORA_Intel/ or script dir"
    exit 1
fi
echo "    using ${PATCH_SCRIPT}"
python "${PATCH_SCRIPT}" "${LLPR_SRC_DIR}"

echo "==> sanity import + patch verification"
python -c "
import mace, inspect, torch
print('mace:', mace.__version__, mace.__file__)
print('torch:', torch.__version__, 'has_xpu_attr:', hasattr(torch, 'xpu'))

from mace.tools import distributed_tools as dt
src = inspect.getsource(dt)
assert 'xccl' in src, 'patch missed: xccl backend'
assert 'PALS_RANKID' in src or 'PMI_RANK' in src, 'patch missed: PALS/PMI env vars'
print('patch verified: xccl + PALS/PMI launcher')

from mace.cli import run_train as rt
src2 = inspect.getsource(rt)
assert 'hasattr(torch, \"xpu\")' in src2, 'patch missed: ipex optional'
print('patch verified: ipex/oneccl optional + xpu device guard')

# LLPR feature present
from mace.modules.llpr import LLPRCache            # noqa: F401
from mace.tools.llpr_features import LastLayerFeatureExtractor  # noqa: F401
print('LLPR present: LLPRCache + LastLayerFeatureExtractor')

# SOAP+FPS deps for the frame assembler
from dscribe.descriptors import SOAP               # noqa: F401
from skmatter.sample_selection import FPS          # noqa: F401
print('assembler deps present: dscribe SOAP + skmatter FPS')
"

echo "Done. To use this venv:"
echo "  source $(cd "$(dirname "$0")" && pwd)/env_aurora.sh"
