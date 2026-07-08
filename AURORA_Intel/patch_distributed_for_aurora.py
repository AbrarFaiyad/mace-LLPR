"""Patch RSGA's mace/tools/distributed_tools.py for Aurora.

Two changes:
  1. xpu code path uses `xccl` backend instead of `ccl`
     (Aurora deprecated ccl in favor of xccl in newer py-torch builds).
  2. mpi launcher also recognises Intel-MPI / PALS env vars
     (PMI_RANK / PALS_RANKID) in addition to OpenMPI's OMPI_COMM_WORLD_RANK.

Idempotent: detects already-patched files via a marker comment and
skips them. Usage:

    python patch_distributed_for_aurora.py <RSGA_repo_dir>/mace
"""
import argparse
import sys
from pathlib import Path

MARKER = "# AURORA-PATCHED"


def patch_file(path: Path) -> bool:
    text = path.read_text()
    if MARKER in text:
        print(f"  [skip] already patched: {path}")
        return False

    new_text = text

    # ── 1. backend ccl -> xccl for the xpu branch ────────────────────────
    if 'backend="ccl"' not in new_text:
        print(f"  [warn] could not find `backend=\"ccl\"` in {path}; "
              f"backend rewrite skipped")
    else:
        new_text = new_text.replace('backend="ccl"', 'backend="xccl"')
        print(f"  [ok]   backend ccl -> xccl")

    # ── 2. mpi launcher: also accept Intel-MPI / PALS ────────────────────
    old_mpi_block = (
        '    elif args.launcher == "mpi":\n'
        '        # OpenMPI & Intel-MPI export these:\n'
        '        rank = int(os.environ["OMPI_COMM_WORLD_RANK"])\n'
        '        world_size = int(os.environ["OMPI_COMM_WORLD_SIZE"])\n'
        '\n'
        '        # local-rank isn’t standardised; compute it from local node-size\n'
        '        local_size = int(os.environ.get("OMPI_COMM_WORLD_LOCAL_SIZE", 1))\n'
        '        local_rank = rank % local_size'
    )
    new_mpi_block = (
        '    elif args.launcher == "mpi":\n'
        '        # OpenMPI exports OMPI_*; Intel MPI / PALS export PMI_*/PALS_*.\n'
        '        if "OMPI_COMM_WORLD_RANK" in os.environ:\n'
        '            rank = int(os.environ["OMPI_COMM_WORLD_RANK"])\n'
        '            world_size = int(os.environ["OMPI_COMM_WORLD_SIZE"])\n'
        '            local_size = int(os.environ.get("OMPI_COMM_WORLD_LOCAL_SIZE", 1))\n'
        '            local_rank = rank % local_size\n'
        '        else:\n'
        '            rank = int(os.environ.get("PMI_RANK",\n'
        '                       os.environ.get("PALS_RANKID", 0)))\n'
        '            # PMI_SIZE = global world size; PALS_NTASKS = same on PALS;\n'
        '            # PALS_LOCAL_SIZE is per-NODE size, NOT a valid world_size.\n'
        '            world_size = int(os.environ.get("PMI_SIZE",\n'
        '                       os.environ.get("PALS_NTASKS",\n'
        '                       os.environ.get("WORLD_SIZE", 1))))\n'
        '            local_rank = int(os.environ.get("PALS_LOCAL_RANKID",\n'
        '                       os.environ.get("MPI_LOCALRANKID", rank)))'
    )

    if old_mpi_block in new_text:
        new_text = new_text.replace(old_mpi_block, new_mpi_block)
        print(f"  [ok]   mpi launcher now accepts PMI_RANK/PALS_RANKID")
    else:
        print(f"  [warn] could not find OMPI mpi block in {path}; "
              f"mpi-block rewrite skipped")

    if new_text == text:
        print(f"  [noop] no changes applied to {path}")
        return False

    header = (
        f"{MARKER} on {__import__('datetime').datetime.now().isoformat(timespec='seconds')}\n"
        f"{MARKER} backend: ccl -> xccl; mpi launcher: + PMI_RANK/PALS_RANKID\n"
    )
    new_text = header + new_text
    path.write_text(new_text)
    print(f"  [WRITE] {path}")
    return True


def patch_run_train_ipex_optional(path: Path) -> bool:
    """Two changes to mace/cli/run_train.py for Aurora:
      A) ipex/oneccl imports made optional when torch.xpu is built-in.
      B) torch.xpu.set_device(local_rank): when ZE_AFFINITY_MASK exposes
         exactly one tile per rank (gpu_tile_compact.sh pattern),
         device_count==1 and local_rank>=1 raises. Use device 0 instead."""
    text = path.read_text()
    if MARKER in text:
        print(f"  [skip] already patched: {path}")
        return False

    # ── (A) ipex import optional ───────────────────────────────────────────
    old_a = (
        '    if args.device == "xpu":\n'
        '        try:\n'
        '            import intel_extension_for_pytorch as ipex\n'
        '            import oneccl_bindings_for_pytorch as oneccl  # pylint: disable=unused-import\n'
        '        except ImportError as e:\n'
        '            raise ImportError(\n'
        '                "Error: Intel extension for PyTorch not found, but XPU device was specified"\n'
        '            ) from e'
    )
    new_a = (
        '    if args.device == "xpu":\n'
        '        # Aurora py-torch 2.10 ships torch.xpu built-in; ipex/oneccl are\n'
        '        # not required there. Only raise if torch lacks xpu support.\n'
        '        try:\n'
        '            import intel_extension_for_pytorch as ipex  # type: ignore  # noqa: F401\n'
        '            import oneccl_bindings_for_pytorch as oneccl  # type: ignore  # noqa: F401\n'
        '        except ImportError:\n'
        '            if not hasattr(torch, "xpu"):\n'
        '                raise ImportError(\n'
        '                    "Error: torch.xpu not found and intel_extension_for_pytorch "\n'
        '                    "is also missing; cannot use --device=xpu"\n'
        '                )'
    )

    # ── (B) set_device safety for gpu_tile_compact.sh ─────────────────────
    old_b = (
        '    if args.distributed:\n'
        '        if args.device == "cuda":\n'
        '            torch.cuda.set_device(local_rank)\n'
        '        elif args.device == "xpu":\n'
        '            torch.xpu.set_device(local_rank)'
    )
    new_b = (
        '    if args.distributed:\n'
        '        if args.device == "cuda":\n'
        '            torch.cuda.set_device(local_rank)\n'
        '        elif args.device == "xpu":\n'
        '            # gpu_tile_compact.sh (Aurora) sets ZE_AFFINITY_MASK so each\n'
        '            # rank sees exactly ONE xpu tile; in that case device 0 is\n'
        '            # the right (and only) choice regardless of local_rank.\n'
        '            try:\n'
        '                n_visible = torch.xpu.device_count()\n'
        '            except Exception:\n'
        '                n_visible = 1\n'
        '            torch.xpu.set_device(local_rank if local_rank < n_visible else 0)'
    )

    # ── (C) ipex.optimize() skipped when ipex not importable ──────────────
    old_c = (
        '    if args.device == "xpu":\n'
        '        logging.info("Optimzing model and optimzier for XPU")\n'
        '        model, optimizer = ipex.optimize(model, optimizer=optimizer)'
    )
    new_c = (
        '    if args.device == "xpu":\n'
        '        try:\n'
        '            import intel_extension_for_pytorch as _ipex  # type: ignore\n'
        '            logging.info("Optimizing model and optimizer with intel_extension_for_pytorch")\n'
        '            model, optimizer = _ipex.optimize(model, optimizer=optimizer)\n'
        '        except ImportError:\n'
        '            logging.info("intel_extension_for_pytorch unavailable; skipping ipex.optimize() (Aurora torch.xpu native path)")'
    )

    # ── (C2) second ipex.optimize site (post-DDP, exact full block) ───────
    old_c2 = (
        '    if args.device == "xpu":\n'
        '        try:\n'
        '            model, optimizer = ipex.optimize(model, optimizer=optimizer)\n'
        '        except ImportError as e:\n'
        '            logging.error(\n'
        '                "Intel Extension for PyTorch not found, but XPU device was specified. "\n'
        '                "Please install it to use XPU device."\n'
        '            )\n'
    )
    new_c2 = (
        '    if args.device == "xpu":\n'
        '        try:\n'
        '            import intel_extension_for_pytorch as _ipex2  # type: ignore\n'
        '            model, optimizer = _ipex2.optimize(model, optimizer=optimizer)\n'
        '        except ImportError:\n'
        '            logging.info(\n'
        '                "intel_extension_for_pytorch unavailable at second optimize "\n'
        '                "site; skipping (Aurora native torch.xpu path)"\n'
        '            )\n'
    )

    # ── (D) DDP device_ids guard for gpu_tile_compact (xpu, 1 tile/rank) ──
    old_d = (
        '    if args.distributed:\n'
        '        distributed_model = DDP(model, device_ids=[local_rank])'
    )
    new_d = (
        '    if args.distributed:\n'
        '        # With ZE_AFFINITY_MASK (Aurora) each rank sees 1 xpu tile;\n'
        '        # local_rank can exceed visible device count, causing DDP to\n'
        '        # raise "value cannot be converted to type int without overflow".\n'
        '        # Pin DDP to device 0 in that case.\n'
        '        _ddp_dev = local_rank\n'
        '        if args.device == "xpu":\n'
        '            try:\n'
        '                if torch.xpu.device_count() == 1:\n'
        '                    _ddp_dev = 0\n'
        '            except Exception:\n'
        '                _ddp_dev = 0\n'
        '        distributed_model = DDP(model, device_ids=[_ddp_dev])'
    )

    new_text = text
    changed = False
    if old_a in new_text:
        new_text = new_text.replace(old_a, new_a)
        print(f"  [ok] ipex import made optional")
        changed = True
    else:
        print(f"  [warn] ipex import block not found")
    if old_b in new_text:
        new_text = new_text.replace(old_b, new_b)
        print(f"  [ok] xpu.set_device guarded against ZE_AFFINITY_MASK")
        changed = True
    else:
        print(f"  [warn] set_device block not found")
    if old_c in new_text:
        new_text = new_text.replace(old_c, new_c)
        print(f"  [ok] ipex.optimize() #1 skipped when ipex not installed")
        changed = True
    else:
        print(f"  [warn] ipex.optimize block #1 not found")
    if old_c2 in new_text:
        new_text = new_text.replace(old_c2, new_c2)
        print(f"  [ok] ipex.optimize() #2 skipped when ipex not installed")
        changed = True
    else:
        print(f"  [warn] ipex.optimize block #2 not found")
    if old_d in new_text:
        new_text = new_text.replace(old_d, new_d)
        print(f"  [ok] DDP device_ids guarded against gpu_tile_compact")
        changed = True
    else:
        print(f"  [warn] DDP construction block not found")
    if not changed:
        return False

    header = (
        f"{MARKER} on {__import__('datetime').datetime.now().isoformat(timespec='seconds')}\n"
        f"{MARKER} ipex/oneccl optional + xpu.set_device guarded for gpu_tile_compact\n"
    )
    new_text = header + new_text
    path.write_text(new_text)
    print(f"  [WRITE] {path}")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mace_pkg_root", type=Path,
                    help="RSGA repo's mace/ dir (the one with setup.cfg)")
    args = ap.parse_args()

    target = args.mace_pkg_root / "mace" / "tools" / "distributed_tools.py"
    if not target.exists():
        sys.exit(f"target file not found: {target}")
    print(f"==> patching {target}")
    patch_file(target)

    target2 = args.mace_pkg_root / "mace" / "cli" / "run_train.py"
    if not target2.exists():
        sys.exit(f"target file not found: {target2}")
    print(f"==> patching {target2} (ipex optional)")
    patch_run_train_ipex_optional(target2)


if __name__ == "__main__":
    main()
