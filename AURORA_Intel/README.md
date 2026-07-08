# AURORA_Intel — running mace-LLPR on ALCF Aurora (Intel PVC / XPU)

Everything needed to build and run this fork on **Aurora**, whose stack differs
from the CUDA/OpenMPI assumptions baked into stock MACE:

| Aurora reality | Stock MACE assumes |
|---|---|
| Intel PVC GPUs, `torch.xpu` (native in torch 2.10) | NVIDIA CUDA |
| distributed backend `xccl` | `ccl` (deprecated on Aurora) / `nccl` |
| PALS launcher (`PMI_RANK` / `PALS_RANKID`, `mpiexec --pmi=pmix`) | OpenMPI (`OMPI_COMM_WORLD_*`) |
| 1 tile/rank via `ZE_AFFINITY_MASK` | many devices visible per rank |
| `intel_extension_for_pytorch` optional | ipex mandatory for XPU |

This fork also adds **LLPR** (last-layer posterior / Mahalanobis uncertainty) on
top of upstream ACEsuit MACE — `LLPRCache`, the `mace_build_llpr_cache` CLI, and
`MACECalculator` integration.

## What the venv is

- **Base:** Aurora `frameworks/2025.3.1` module (Python 3.12, XPU-enabled
  PyTorch 2.10, native `xccl`). The venv is created with
  `--system-site-packages` so the vendor torch / numpy / scipy / matplotlib are
  inherited, never reinstalled.
- **MACE:** editable (`pip install -e .`), `--no-deps`, from this fork — so a
  second torch is never pulled in on top of the Aurora one.
- **Extra leaf deps:** `dscribe 2.1.2`, `skmatter 0.3.3` (SOAP + FPS for the
  frame assembler), `ase`, `e3nn 0.4.4`, plus the usual MACE runtime deps.

Works out of the box: `build_env_llpr_aurora.sh` auto-detects this fork's root
(it lives in `AURORA_Intel/`) and defaults the venv to `$HOME/venv_mace_lora_llpr`.
Override either via env var (`VENV_DIR=...`, `LLPR_SRC_DIR=...`) if you want them
elsewhere.

## Files

- **`build_env_llpr_aurora.sh`** — one-shot build of the LLPR pipeline venv.
  frameworks/2025.3.1 module → `--system-site-packages` venv → editable
  `--no-deps` install of this fork → e3nn 0.4.4 → SOAP+FPS deps
  (`dscribe`, `skmatter`) → applies the patch (idempotent) → verifies imports,
  LLPR, and the patch markers. **Edit `VENV_DIR` / `LLPR_SRC_DIR` at the top
  for your install.**
- **`build_rsga_env_aurora.sh`** — sibling build for a separate RSGA training
  venv (a different MACE fork). Same Aurora patch recipe; kept for reference.
- **`patch_distributed_for_aurora.py`** — post-install patcher. Find-and-replaces
  known code blocks in the MACE source (no new modules added; idempotent via a
  `# AURORA-PATCHED` marker; leaves a `.bak-<date>` of each original). Two files,
  6 edits:
  - `mace/tools/distributed_tools.py`: `ccl`→`xccl`; MPI launcher also reads
    `PMI_RANK` / `PALS_RANKID` / `PMI_SIZE` / `PALS_NTASKS` / `PALS_LOCAL_RANKID`.
  - `mace/cli/run_train.py`: ipex/oneccl imports optional; `torch.xpu.set_device`
    and `DDP(device_ids=...)` fall back to device 0 when only one tile is visible;
    both `ipex.optimize()` sites wrapped in try/except.
  Run: `python patch_distributed_for_aurora.py <fork-root>`
- **`env_aurora.sh`** — per-job runtime env: `module load frameworks/2025.3.1`,
  activate the venv, set `ONEAPI_DEVICE_SELECTOR`, `ZE_FLAT_DEVICE_HIERARCHY=FLAT`,
  `ZE_ENABLE_PCI_ID_DEVICE_ORDER`, `MPICH_GPU_SUPPORT_ENABLED`, `VENV_PYTHON`.
  **Edit the venv path inside for your install.**

## Quick start (Aurora login node)

```bash
# edit VENV_DIR / LLPR_SRC_DIR at the top of the script first
bash build_env_llpr_aurora.sh          # build the venv (~few min)
source env_aurora.sh                    # activate for a run
```

## Reproduce from scratch (manual, if you don't use the script)

```bash
# from this fork's root ($HOME/mace-LLPR or wherever you cloned it)
VENV_DIR="$HOME/venv_mace_lora_llpr"

# 1. Aurora framework module (Python 3.12 + XPU torch 2.10 + native xccl)
module load frameworks/2025.3.1

# 2. venv inheriting the vendor torch/numpy/scipy
python -m venv --system-site-packages "$VENV_DIR"
source "$VENV_DIR/bin/activate"

# 3. install this fork editable, no-deps (do NOT pull a second torch)
git checkout main            # patches are baked in on main (see below)
pip install --no-deps -e .
pip install --no-deps "e3nn==0.4.4"

# 4. leaf deps (SOAP+FPS for the frame assembler + MACE runtime deps)
pip install --no-cache-dir dscribe skmatter ase torchmetrics torch-ema \
    matscipy opt-einsum-fx prettytable python-hostlist configargparse \
    h5py tqdm lmdb orjson pandas

# 5. (only needed against an UNPATCHED tree, e.g. a fresh upstream MACE)
python AURORA_Intel/patch_distributed_for_aurora.py .
```

## Verify

```bash
python -c "import dscribe, skmatter, sklearn, scipy, numpy, ase; \
from dscribe.descriptors import SOAP; from skmatter.sample_selection import FPS; \
from mace.modules.llpr import LLPRCache; import mace; \
print('mace', mace.__version__, mace.__file__)"
# expect: imports OK; mace path under this fork
```

## Where each Aurora change lives

- **Baked into this fork's `main`** (a fresh clone runs on Aurora with no patch
  step): the `xccl`/PALS launcher edits in `distributed_tools.py`, the
  ipex-optional + single-tile `set_device`/DDP guards in `run_train.py`, the
  runtime distributed-backend selection, the ipex.optimize XPU inference fix,
  and the whole LLPR feature.
- **`patch_distributed_for_aurora.py` is kept for re-application** — it is
  idempotent (skips files already carrying the `# AURORA-PATCHED` marker), so
  running it on `main` is a no-op. Its value is applying the same edits to a
  *fresh upstream MACE* or a different fork.

If upstream MACE refactors a targeted block, the patch prints `[warn] ... not
found` and no-ops that edit (safe) — update the exact-match strings in the script.
