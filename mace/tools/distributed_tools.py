# AURORA-PATCHED on 2026-06-26T16:00:58
# AURORA-PATCHED backend: ccl -> xccl; mpi launcher: + PMI_RANK/PALS_RANKID
import os

import torch


def init_distributed(args):
    """
    Returns (rank, local_rank, world_size) and initialises the process-group.
    Works for: slurm | torchrun | mpi | none
    """
    if not args.distributed:
        return 0, 0, 1  # single-GPU / debug run

    # ------------------------------------------------------------------ slurm
    if args.launcher == "slurm":
        from mace.tools.slurm_distributed import DistributedEnvironment

        env = DistributedEnvironment()
        rank, local_rank, world_size = env.rank, env.local_rank, env.world_size

    # ---------------------------------------------------------------- torchrun
    elif args.launcher == "torchrun":
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])

    # -------------------------------------------------------------------- mpi
    elif args.launcher == "mpi":
        # OpenMPI exports OMPI_*; Intel MPI / PALS export PMI_*/PALS_*.
        if "OMPI_COMM_WORLD_RANK" in os.environ:
            rank = int(os.environ["OMPI_COMM_WORLD_RANK"])
            world_size = int(os.environ["OMPI_COMM_WORLD_SIZE"])
            local_size = int(os.environ.get("OMPI_COMM_WORLD_LOCAL_SIZE", 1))
            local_rank = rank % local_size
        else:
            rank = int(os.environ.get("PMI_RANK",
                       os.environ.get("PALS_RANKID", 0)))
            # PMI_SIZE = global world size; PALS_NTASKS = same on PALS;
            # PALS_LOCAL_SIZE is per-NODE size, NOT a valid world_size.
            world_size = int(os.environ.get("PMI_SIZE",
                       os.environ.get("PALS_NTASKS",
                       os.environ.get("WORLD_SIZE", 1))))
            local_rank = int(os.environ.get("PALS_LOCAL_RANKID",
                       os.environ.get("MPI_LOCALRANKID", rank)))

        # tell PyTorch where the rendez-vous server is
        os.environ.setdefault("MASTER_ADDR", os.environ["MASTER_ADDR"])
        os.environ.setdefault("MASTER_PORT", os.environ.get("MASTER_PORT", "33333"))
        # torchrun style vars so later code keeps working
        os.environ["RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = str(world_size)
        os.environ["LOCAL_RANK"] = str(local_rank)

    else:  # "none"
        return 0, 0, 1

    if not torch.distributed.is_initialized():
        # AURORA-PATCHED 2026-07-07: stderr instrumentation + bounded timeout.
        # setup_logger() runs AFTER this in run_train, so a hang here would
        # otherwise leave NO log at all. Print to stderr (-> pbs_out) around
        # the rendezvous so we can see which ranks entered vs completed, and
        # cap the TCPStore wait so a bad rank-0 draw fails fast instead of
        # burning the full 30-min default before the chain can retry.
        import datetime
        import socket
        import sys

        _host = socket.gethostname()
        _pg_timeout = datetime.timedelta(
            seconds=int(os.environ.get("MACE_PG_TIMEOUT_SEC", "600"))
        )
        print(
            f"[PGINIT-ENTER] rank={rank} local_rank={local_rank} "
            f"world={world_size} host={_host} "
            f"master={os.environ.get('MASTER_ADDR')}:{os.environ.get('MASTER_PORT')} "
            f"timeout={_pg_timeout}",
            file=sys.stderr,
            flush=True,
        )
        if args.device == "cuda":
            torch.distributed.init_process_group(
                backend="nccl",
                init_method="env://",
                timeout=_pg_timeout,
            )
        elif args.device == "xpu":
            torch.distributed.init_process_group(
                backend="xccl",
                init_method="env://",
                timeout=_pg_timeout,
            )
        print(
            f"[PGINIT-DONE] rank={rank} host={_host}",
            file=sys.stderr,
            flush=True,
        )
    return rank, local_rank, world_size
