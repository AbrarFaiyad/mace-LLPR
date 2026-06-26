###########################################################################################
# CLI: build LLPR (Last-Layer Posterior Regression) uncertainty cache for a MACE model
# Based on Bigi et al., MLST 5(4) 2024, arXiv:2403.02251
# This program is distributed under the MIT License (see MIT.md)
###########################################################################################
"""mace_build_llpr_cache

Build a per-model LLPR uncertainty cache from a trained MACE checkpoint plus a
training-set extxyz file. The resulting cache (.pt) can then be passed to
MACECalculator(llpr_cache_path=...) to get per-atom uncertainties at inference.

Example:

  mace_build_llpr_cache \\
      --model my_trained.model \\
      --train_file train.xyz \\
      --valid_file valid.xyz \\
      --output llpr_cache.pt \\
      --hook_layer pre_last \\
      --calibration spearman \\
      --energy_key REF_energy \\
      --forces_key REF_forces

Supported calibration modes:
  - "auto_pd"  : auto-select smallest regularizer giving stable Cholesky.
                 No validation set needed.
  - "spearman" : sweep regularizer grid, pick value maximizing Spearman
                 correlation between predicted u and reference force error
                 on the validation set. Requires --valid_file with reference
                 forces.
  - "nll"      : auto-PD regularizer + closed-form alpha calibration from
                 squared energy residuals on the validation set. Requires
                 --valid_file with reference energies.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch

from mace.calculators.mace import MACECalculator
from mace.modules.llpr import LLPRCache
from mace.tools.llpr_features import VALID_HOOK_LAYERS, LastLayerFeatureExtractor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build LLPR (Last-Layer Posterior Regression) uncertainty cache "
            "for a trained MACE model. See Bigi et al., MLST 5(4) 2024, "
            "arXiv:2403.02251."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        required=True,
        help="path to trained MACE model file (.model or .pt)",
    )
    parser.add_argument(
        "--train_file",
        required=True,
        help="path to extxyz file with training configurations",
    )
    parser.add_argument(
        "--valid_file",
        default=None,
        help=(
            "path to extxyz file with validation configurations. "
            "Required for --calibration spearman or nll."
        ),
    )
    parser.add_argument(
        "--output",
        required=True,
        help="output path for the LLPR cache (.pt file)",
    )
    parser.add_argument(
        "--hook_layer",
        choices=list(VALID_HOOK_LAYERS),
        default="pre_last",
        help=(
            "Which last-layer linear to hook for feature extraction. "
            "'pre_last' = linear_1 input (richer; d=128 for MATPES). "
            "'auto' = last linear input (d=16 for MATPES NonLinear readout)."
        ),
    )
    parser.add_argument(
        "--calibration",
        choices=["auto_pd", "spearman", "nll"],
        default="auto_pd",
        help=(
            "Calibration mode. 'auto_pd' needs no validation set. "
            "'spearman'/'nll' require --valid_file."
        ),
    )
    parser.add_argument(
        "--lambda_sq_grid",
        nargs="+",
        type=float,
        default=[1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0],
        help="regularizer grid for --calibration spearman (log-spaced suggested)",
    )
    parser.add_argument(
        "--energy_key",
        default="REF_energy",
        help="key in extxyz frame.info holding the reference energy",
    )
    parser.add_argument(
        "--forces_key",
        default="REF_forces",
        help="key in extxyz frame.arrays holding the reference forces",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda", "xpu", "mps"],
        default="auto",
        help="torch device; 'auto' picks cuda > xpu > mps > cpu",
    )
    parser.add_argument(
        "--default_dtype",
        choices=["float32", "float64"],
        default="float64",
        help="model dtype (fp64 strongly recommended for LLPR numerics)",
    )
    parser.add_argument(
        "--num_frames_train",
        type=int,
        default=None,
        help="(optional) cap training frames used for covariance build",
    )
    parser.add_argument(
        "--num_frames_valid",
        type=int,
        default=None,
        help="(optional) cap validation frames used for calibration",
    )
    parser.add_argument(
        "--log_level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
    )
    return parser.parse_args()


def resolve_device(arg: str) -> str:
    if arg != "auto":
        return arg
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return "xpu"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level),
        format="[%(asctime)s][LLPR][%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )


def load_model_for_inference(model_path: str, device: str, dtype: str) -> MACECalculator:
    """Load a MACE model checkpoint via CPU roundtrip to avoid map_location quirks
    on XPU / older CUDA tags. Returns a ready-to-use MACECalculator."""
    logging.info("Loading model from %s (cpu-roundtrip, device=%s)", model_path, device)
    loaded = torch.load(str(model_path), map_location="cpu", weights_only=False)
    if not isinstance(loaded, list):
        loaded = [loaded]
    return MACECalculator(
        models=loaded, device=device, default_dtype=dtype
    )


def read_extxyz(path: str, n_cap: Optional[int]) -> List:
    """Read extxyz frames; optional cap on count (for fast smoke tests)."""
    from ase.io import read as ase_read

    frames = ase_read(path, index=":", format="extxyz")
    if n_cap is not None and len(frames) > n_cap:
        logging.info("Capping frames from %d to %d", len(frames), n_cap)
        frames = frames[:n_cap]
    return frames


def collect_features(
    frames: List,
    calc: MACECalculator,
    extractor: LastLayerFeatureExtractor,
    label: str,
) -> torch.Tensor:
    """Run forward on every frame, accumulate captured features into a tensor.

    Forces compute_force=False where supported so that no backward path runs
    inside torch.no_grad() (which would throw on most MACE versions).
    """
    feats: List[torch.Tensor] = []
    model = calc.models[0]
    n = len(frames)
    progress_stride = max(n // 10, 1)
    with torch.no_grad():
        for i, atoms in enumerate(frames):
            batch = calc._atoms_to_batch(atoms)
            batch_dict = batch.to_dict()
            try:
                model(batch_dict, compute_force=False)
            except TypeError:
                # older MACE: forward() doesn't take compute_force kwarg
                model(batch_dict)
            phi = extractor.last_captured().detach().double().cpu()
            feats.append(phi)
            if (i + 1) % progress_stride == 0 or (i + 1) == n:
                logging.info("  [%s] %d / %d frames", label, i + 1, n)
    return torch.cat(feats, dim=0)


def compute_per_atom_force_error(
    frames: List,
    calc: MACECalculator,
    forces_key: str,
) -> torch.Tensor:
    """Compute |F_pred - F_ref| per atom over frames; return flat tensor."""
    errors: List[torch.Tensor] = []
    for atoms in frames:
        if forces_key not in atoms.arrays:
            raise KeyError(
                f"reference forces key '{forces_key}' not found in frame; "
                f"available keys: {list(atoms.arrays.keys())}"
            )
        ref = torch.tensor(atoms.arrays[forces_key], dtype=torch.float64)
        at_eval = atoms.copy()
        at_eval.calc = calc
        pred = torch.tensor(at_eval.get_forces(), dtype=torch.float64)
        errors.append((pred - ref).norm(dim=-1))
    return torch.cat(errors, dim=0)


def compute_per_frame_energy_residual(
    frames: List,
    calc: MACECalculator,
    energy_key: str,
) -> torch.Tensor:
    """E_pred - E_ref per frame."""
    residuals: List[float] = []
    for atoms in frames:
        if energy_key not in atoms.info:
            raise KeyError(
                f"reference energy key '{energy_key}' not found in frame.info; "
                f"available: {list(atoms.info.keys())}"
            )
        ref = float(atoms.info[energy_key])
        at_eval = atoms.copy()
        at_eval.calc = calc
        pred = float(at_eval.get_potential_energy())
        residuals.append(pred - ref)
    return torch.tensor(residuals, dtype=torch.float64)


def baseline_norm_u_spearman(
    phi_valid: torch.Tensor,
    F_err_valid: torch.Tensor,
) -> Optional[float]:
    """Quick baseline: Spearman( ||phi||_2 , F_err ) for reporting."""
    try:
        from scipy.stats import spearmanr
    except ImportError:
        return None
    u_norm = phi_valid.norm(dim=-1).cpu().numpy()
    rho, _ = spearmanr(u_norm, F_err_valid.cpu().numpy())
    return float(rho) if rho is not None and not np.isnan(rho) else None


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)

    device = resolve_device(args.device)
    logging.info("Resolved device: %s", device)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if args.calibration in ("spearman", "nll") and args.valid_file is None:
        sys.exit(f"--calibration {args.calibration} requires --valid_file")

    # ---- load model + attach hook ----
    calc = load_model_for_inference(args.model, device, args.default_dtype)
    model = calc.models[0]
    extractor = LastLayerFeatureExtractor(model, hook_layer=args.hook_layer)
    logging.info(
        "Hook attached: hook_layer=%s, feature dim d=%d",
        args.hook_layer,
        extractor.d,
    )

    # ---- accumulate training features ----
    train_frames = read_extxyz(args.train_file, args.num_frames_train)
    logging.info(
        "Accumulating Phi^T Phi from %d training frames (%d total atoms)",
        len(train_frames),
        sum(len(at) for at in train_frames),
    )
    cache = LLPRCache(d=extractor.d, device=device)
    phi_train_chunks: List[torch.Tensor] = []
    model = calc.models[0]
    with torch.no_grad():
        for i, atoms in enumerate(train_frames):
            batch = calc._atoms_to_batch(atoms)
            batch_dict = batch.to_dict()
            try:
                model(batch_dict, compute_force=False)
            except TypeError:
                model(batch_dict)
            phi = extractor.last_captured().detach().double()
            cache.accumulate(phi)
            phi_train_chunks.append(phi.cpu())
            if (i + 1) % max(len(train_frames) // 10, 1) == 0 or (i + 1) == len(train_frames):
                logging.info("  [train] %d / %d frames", i + 1, len(train_frames))

    # ---- calibration ----
    sweep_results: Optional[List] = None
    baseline_rho: Optional[float] = None

    if args.calibration == "auto_pd":
        logging.info("Calibration: auto_pd (no validation needed)")
        cache.finalize(regularizer_sq=None)
        logging.info(
            "Auto-PD chose regularizer_sq = %.4g (smallest stable Cholesky)",
            cache.regularizer_sq,
        )
    elif args.calibration == "spearman":
        valid_frames = read_extxyz(args.valid_file, args.num_frames_valid)
        logging.info(
            "Collecting valid features + force error over %d valid frames",
            len(valid_frames),
        )
        phi_valid = collect_features(valid_frames, calc, extractor, "valid")
        F_err_valid = compute_per_atom_force_error(
            valid_frames, calc, args.forces_key
        )
        baseline_rho = baseline_norm_u_spearman(phi_valid, F_err_valid)
        if baseline_rho is not None:
            logging.info("Baseline norm-u Spearman: %.4f", baseline_rho)

        logging.info(
            "Sweeping %d regularizer values: %s",
            len(args.lambda_sq_grid),
            args.lambda_sq_grid,
        )
        best_lam, best_rho, sweep_results = cache.calibrate_spearman(
            phi_valid, F_err_valid, lambdas_sq=args.lambda_sq_grid
        )
        logging.info(
            "Best regularizer_sq = %.4g, Spearman = %.4f",
            best_lam,
            best_rho,
        )
        for lam, rho in sweep_results:
            marker = " <-- BEST" if lam == best_lam else ""
            logging.info("  reg=%-12.4g  rho=%.4f%s", lam, rho, marker)
    elif args.calibration == "nll":
        logging.info("Calibration: auto-PD regularizer + NLL alpha calibration")
        cache.finalize(regularizer_sq=None)
        logging.info(
            "Auto-PD chose regularizer_sq = %.4g", cache.regularizer_sq
        )
        valid_frames = read_extxyz(args.valid_file, args.num_frames_valid)
        # NLL calibration needs per-frame uncalibrated u^2; build feature pool
        # by summing per-atom phi within each frame (energy is a sum of atomic
        # contributions, so phi sums match).
        per_frame_phi: List[torch.Tensor] = []
        for atoms in valid_frames:
            batch = calc._atoms_to_batch(atoms)
            batch_dict = batch.to_dict()
            try:
                model(batch_dict, compute_force=False)
            except TypeError:
                model(batch_dict)
            phi_atoms = extractor.last_captured().detach().double().cpu()
            per_frame_phi.append(phi_atoms.sum(dim=0, keepdim=True))
        phi_frames = torch.cat(per_frame_phi, dim=0)
        residuals = compute_per_frame_energy_residual(
            valid_frames, calc, args.energy_key
        )
        alpha_sq = cache.calibrate_nll(phi_frames, residuals)
        logging.info("NLL calibration: alpha_sq = %.4g", alpha_sq)
    else:
        raise RuntimeError(f"unknown calibration={args.calibration!r}")

    # ---- save ----
    cache.save(out_path)
    logging.info("Saved LLPR cache to %s", out_path)
    logging.info(
        "  d=%d, regularizer_sq=%.4g, alpha_sq=%.4g",
        cache.d,
        cache.regularizer_sq,
        cache.alpha_sq,
    )

    # ---- write manifest sidecar ----
    manifest_path = out_path.with_suffix(out_path.suffix + ".manifest.json")
    import json as _json

    manifest = {
        "schema_version": 1,
        "model_path": str(Path(args.model).resolve()),
        "train_file": str(Path(args.train_file).resolve()),
        "valid_file": (
            str(Path(args.valid_file).resolve()) if args.valid_file else None
        ),
        "hook_layer": args.hook_layer,
        "calibration": args.calibration,
        "d": cache.d,
        "regularizer_sq": float(cache.regularizer_sq),
        "alpha_sq": float(cache.alpha_sq),
        "n_train_frames": len(train_frames),
        "n_train_atoms": int(sum(len(at) for at in train_frames)),
        "device": device,
        "default_dtype": args.default_dtype,
        "baseline_norm_u_spearman": baseline_rho,
        "sweep_results": (
            [{"regularizer_sq": float(l), "spearman": float(r)}
             for l, r in sweep_results]
            if sweep_results is not None
            else None
        ),
    }
    manifest_path.write_text(_json.dumps(manifest, indent=2) + "\n")
    logging.info("Wrote manifest to %s", manifest_path)


if __name__ == "__main__":
    main()
