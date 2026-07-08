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

## Files

- **`patch_distributed_for_aurora.py`** — post-install patcher. Find-and-replaces
  known code blocks in the *installed* MACE (no new modules added; idempotent via
  a `# AURORA-PATCHED` marker). Two files, 6 edits:
  - `mace/tools/distributed_tools.py`: `ccl`→`xccl`; MPI launcher also reads
    `PMI_RANK`/`PALS_RANKID`/`PMI_SIZE`/`PALS_NTASKS`/`PALS_LOCAL_RANKID`.
  - `mace/cli/run_train.py`: ipex/oneccl imports optional; `torch.xpu.set_device`
    and `DDP(device_ids=...)` fall back to device 0 when only one tile is visible;
    both `ipex.optimize()` sites wrapped in try/except.
  Run: `python patch_distributed_for_aurora.py <path-to-fork-root>`
- **`build_env_llpr_aurora.sh`** — one-shot build of the LLPR pipeline venv
  (`venv_mace_lora_llpr`): frameworks/2025.3.1 module → `--system-site-packages`
  venv → editable `--no-deps` install of this fork → e3nn 0.4.4 → SOAP+FPS deps
  (`dscribe`, `skmatter`) → applies the patch → verifies imports + LLPR + patches.
- **`build_rsga_env_aurora.sh`** — sibling build for the separate RSGA training
  venv (`venv_mace_rsga`, GSLab2025/RSGA_MACE_OPT). Same Aurora patch recipe;
  kept here for reference.
- **`env_aurora.sh`** — per-job runtime env: `module load frameworks/2025.3.1`,
  activate the venv, set `ONEAPI_DEVICE_SELECTOR`, `ZE_FLAT_DEVICE_HIERARCHY=FLAT`,
  `ZE_ENABLE_PCI_ID_DEVICE_ORDER`, `MPICH_GPU_SUPPORT_ENABLED`, `VENV_PYTHON`.
- **`AURORA_LLPR_VENV_BUILD.md`** — narrative provenance + reproduce-from-scratch
  commands + which Aurora changes are committed to the fork vs applied by the
  patch script.

## Quick start (login node)

```bash
bash build_env_llpr_aurora.sh          # build the venv (~few min)
source env_aurora.sh                    # activate for a run
```

## Committed vs patched — where each Aurora change lives

- **Committed to this fork** (survives a fresh clone, no patch needed):
  runtime distributed-backend selection (`distributed_tools.py`, commit
  `64c2309`) and the ipex.optimize XPU inference fix (`calculators/mace.py`,
  commit `9c9858a`), plus the whole LLPR feature.
- **Applied by `patch_distributed_for_aurora.py`** (launcher/environment glue
  that is brittle to rebase, so kept replayable): the `xccl` rename, the
  PALS/PMI launcher branch, and the ipex-optional / single-tile device guards in
  `run_train.py`.

If upstream MACE refactors a targeted block, the patch prints `[warn] ... not
found` and no-ops that edit (safe) — update the exact-match strings in the script.
