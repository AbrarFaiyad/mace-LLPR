#!/bin/bash
# Aurora env for MATPES LoRA finetune pipeline.
#
# Uses venv_mace_lora_llpr (editable mace-LLPR fork with first-class LLPR
# uncertainty support, plus Aurora xccl + ipex + DDP patches).
#
# Built on frameworks/2025.3.1: torch 2.10 + ipex 2.10 + native xccl XPU
# collective (no oneccl wheel needed).

# VENV_DIR must match what build_env_llpr_aurora.sh created. Override by
# exporting VENV_DIR before sourcing; default is $HOME/venv_mace_lora_llpr.
VENV_DIR="${VENV_DIR:-$HOME/venv_mace_lora_llpr}"

# Source lmod + load frameworks/2025.3.1 (default Aurora stack)
source /usr/share/lmod/lmod/init/bash 2>/dev/null
module load frameworks/2025.3.1 2>/dev/null

# Activate venv: editable mace-LLPR (Aurora-patched) inheriting torch+ipex from frameworks
source "${VENV_DIR}/bin/activate"

# Aurora XPU runtime essentials
export ONEAPI_DEVICE_SELECTOR="${ONEAPI_DEVICE_SELECTOR:-level_zero:gpu}"
export ZE_FLAT_DEVICE_HIERARCHY="${ZE_FLAT_DEVICE_HIERARCHY:-FLAT}"
export ZE_ENABLE_PCI_ID_DEVICE_ORDER=1
export MPICH_GPU_SUPPORT_ENABLED=1
ulimit -c 0

export VENV_PYTHON="${VENV_DIR}/bin/python3"
