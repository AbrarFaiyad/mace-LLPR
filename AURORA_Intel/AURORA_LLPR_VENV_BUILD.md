# Building the Aurora LLPR venv (`venv_mace_lora_llpr`)

Handoff note. Reproduces the MACE-LLPR training/inference environment used by the
Auto-Finetuner active-learning pipeline on ALCF **Aurora** (Intel PVC / XPU).

## What it is

- **Location:** `/lus/flare/projects/QuantumDS/afaiyad/venv_mace_lora_llpr`
- **Base:** Aurora `frameworks/2025.3.1` module (Python 3.12.12, XPU-enabled
  PyTorch 2.10) — created with `--system-site-packages` so the vendor torch /
  numpy / scipy / matplotlib are inherited, never reinstalled.
- **MACE:** editable install of our fork **mace-LLPR**
  (`git@github.com:AbrarFaiyad/mace-LLPR.git`, branch `main`) at
  `/lus/flare/projects/QuantumDS/afaiyad/mace-LLPR-fork`. This fork adds the
  LLPR (last-layer posterior / Mahalanobis uncertainty) implementation on top of
  upstream ACEsuit MACE.
- **Extra leaf deps** (pip, no torch clobber): `dscribe 2.1.2`, `skmatter 0.3.3`
  (SOAP + FPS for the frame assembler), `ase`, `e3nn 0.4.4`.

## Provenance of the Aurora-specific changes

Aurora needs adaptations that stock MACE does not ship. They live in **two
layers**:

### A. Committed to the mace-LLPR fork
- `mace/tools/distributed_tools.py` — commit `64c2309`
  *"dynamically set backend for distributed training based on device type"*
  (picks the XPU distributed backend at runtime).
- `mace/calculators/mace.py` — commit `9c9858a`
  *"fix(calculator): ipex.optimize inference path for XPU"* (the inference-mode
  fix so the LLPR calculator runs on XPU without the pre-existing ipex crash).
- The LLPR feature itself: commits `a582826` (core cache + hook), `467bc3a`
  (`mace_build_llpr_cache` CLI), `f1f025d` (MACECalculator integration),
  `482d1a0` (docs/example).

### B. Applied post-install by a patch script (NOT upstream, NOT in the fork commits)
`patch_distributed_for_aurora.py` (in the Auto-Finetuner repo root) rewrites
`site-packages`/editable `mace/tools/distributed_tools.py` for Aurora:
1. **`backend="ccl"` → `backend="xccl"`** — Aurora deprecated Intel `ccl` in
   favour of `xccl` in newer py-torch builds.
2. **MPI launcher also honours Intel-MPI / PALS env vars** (`PMI_RANK`,
   `PALS_RANKID`) in addition to OpenMPI's `OMPI_COMM_WORLD_RANK` — required
   because Aurora launches via PALS/`mpiexec --pmi=pmix`, not OpenMPI.
The script is **idempotent** (marker `# AURORA-PATCHED`) and leaves a
`.bak-<date>` of the original. It also makes the `ipex`/`oneccl` imports optional
in `mace/cli/run_train.py` so training does not hard-fail when ipex is absent.

> Note: the repo's `build_env_aurora.sh` builds a *different* venv
> (`venv_mace_rsga`, the RSGA fork) but uses the **same pattern** — it is the
> reference for the module loads + patch invocation. The LLPR venv was built by
> hand this session following that same recipe against the mace-LLPR fork.

## Reproduce from scratch

```bash
# 1. login node, load the Aurora framework module
module load frameworks/2025.3.1        # Python 3.12 + XPU torch 2.10

# 2. create venv inheriting the vendor torch/numpy/scipy
python -m venv --system-site-packages \
    /lus/flare/projects/QuantumDS/afaiyad/venv_mace_lora_llpr
source /lus/flare/projects/QuantumDS/afaiyad/venv_mace_lora_llpr/bin/activate

# 3. clone the LLPR fork
git clone git@github.com:AbrarFaiyad/mace-LLPR.git \
    /lus/flare/projects/QuantumDS/afaiyad/mace-LLPR-fork
cd /lus/flare/projects/QuantumDS/afaiyad/mace-LLPR-fork
git checkout main            # includes the committed XPU-backend + ipex fixes

# 4. install MACE editable, no-deps (do NOT pull a second torch)
pip install --no-deps -e .
pip install --no-deps "e3nn==0.4.4"

# 5. leaf deps (SOAP+FPS for the frame assembler, + MACE runtime deps)
pip install --no-cache-dir dscribe skmatter ase torchmetrics torch-ema \
    matscipy opt-einsum-fx prettytable python-hostlist configargparse \
    h5py tqdm lmdb orjson pandas

# 6. apply the Aurora ccl->xccl + PALS-launcher patch (idempotent)
python /lus/flare/projects/QuantumDS/afaiyad/Auto-Finetuner/patch_distributed_for_aurora.py \
    /lus/flare/projects/QuantumDS/afaiyad/mace-LLPR-fork/mace
```

## Verify

```bash
python -c "import dscribe, skmatter, sklearn, scipy, numpy, ase; \
from dscribe.descriptors import SOAP; from skmatter.sample_selection import FPS; \
import mace; print('mace', mace.__version__, mace.__file__)"
# expect: imports OK; mace path under mace-LLPR-fork
```

At time of writing the fork is at commit `9c9858a` (main).

## Runtime env (per-job)
Activation + XPU env for actual runs is in `env_aurora.sh` (module restore +
`ZE_FLAT_DEVICE_HIERARCHY=FLAT`, per-tile `ZE_AFFINITY_MASK`, `--pmi=pmix`
launcher). See `al_lib/md_orchestrator.py` for the per-rank tile-pinning wrapper.
