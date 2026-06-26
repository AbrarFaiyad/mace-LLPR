"""End-to-end LLPR uncertainty quickstart using the bundled MACE-MP foundation.

This example shows the full workflow:
  1. Load a MACE foundation model (bundled MACE-MP, 44 MB)
  2. Build a small synthetic training+validation dataset (bulk Si)
  3. Build an LLPR uncertainty cache via the Python API (mirrors what
     mace_build_llpr_cache CLI does)
  4. Plug the cache into MACECalculator and read per-atom uncertainty
     from the standard ASE results dict

Runs on CPU in <30 seconds. No GPU required.

Run:
    python examples/llpr_quickstart.py
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
from ase.build import bulk

from mace.calculators.mace import MACECalculator
from mace.modules.llpr import LLPRCache
from mace.tools.llpr_features import LastLayerFeatureExtractor


def build_synthetic_dataset(calc: MACECalculator, n_train: int, n_valid: int):
    """Use the foundation as 'reference' to self-label distorted Si frames.
    Produces train_frames, valid_frames lists of ase.Atoms with REF_forces."""
    rng = np.random.RandomState(0)
    frames = []
    for k in range(n_train + n_valid):
        atoms = bulk(
            "Si",
            cubic=True,
            a=5.43 + 0.05 * rng.randn(),
            crystalstructure="diamond",
        )
        atoms.rattle(stdev=0.03 + 0.02 * (k / (n_train + n_valid)),
                     seed=rng.randint(1 << 30))
        atoms.calc = calc
        F = atoms.get_forces()
        atoms.arrays["REF_forces"] = F
        atoms.info["REF_energy"] = atoms.get_potential_energy()
        frames.append(atoms)
    return frames[:n_train], frames[n_train:]


def build_cache(
    model_path: str,
    train_frames,
    valid_frames,
    device: str,
    hook_layer: str,
) -> Path:
    """Build an LLPR cache and save it to a tmp file. Returns the cache path."""
    loaded = torch.load(model_path, map_location="cpu", weights_only=False)
    calc = MACECalculator(
        models=[loaded], device=device, default_dtype="float64"
    )
    model = calc.models[0]
    extractor = LastLayerFeatureExtractor(model, hook_layer=hook_layer)
    cache = LLPRCache(d=extractor.d, device=device)
    print(f"[llpr] hook_layer={hook_layer}, feature dim d={extractor.d}")

    # accumulate covariance from training frames
    print(f"[llpr] accumulating covariance from {len(train_frames)} training frames")
    with torch.no_grad():
        for atoms in train_frames:
            batch = calc._atoms_to_batch(atoms).to_dict()
            try:
                model(batch, compute_force=False)
            except TypeError:
                model(batch)
            cache.accumulate(extractor.last_captured().double())

    # spearman calibration: collect valid features + per-atom force error
    print(f"[llpr] computing per-atom force error on {len(valid_frames)} valid frames")
    phi_valid_chunks = []
    F_err_chunks = []
    for atoms in valid_frames:
        batch = calc._atoms_to_batch(atoms).to_dict()
        with torch.no_grad():
            try:
                model(batch, compute_force=False)
            except TypeError:
                model(batch)
        phi_valid_chunks.append(extractor.last_captured().detach().double().cpu())

        ref = torch.tensor(atoms.arrays["REF_forces"], dtype=torch.float64)
        at_eval = atoms.copy()
        at_eval.calc = calc
        pred = torch.tensor(at_eval.get_forces(), dtype=torch.float64)
        F_err_chunks.append((pred - ref).norm(dim=-1))
    phi_valid = torch.cat(phi_valid_chunks, dim=0)
    F_err = torch.cat(F_err_chunks, dim=0)
    print(f"[llpr] sweeping regularizer to maximize Spearman(u, |F_err|) on valid")
    best_lam, best_rho, sweep = cache.calibrate_spearman(
        phi_valid, F_err,
        lambdas_sq=(1e-6, 1e-4, 1e-2, 1.0, 10.0, 100.0),
    )
    print(f"[llpr] best regularizer_sq={best_lam:.4g}, Spearman ρ={best_rho:.4f}")

    cache_path = Path(tempfile.gettempdir()) / "mace_llpr_quickstart_cache.pt"
    cache.save(cache_path)
    extractor.remove()
    print(f"[llpr] cache saved to {cache_path}")
    return cache_path


def inference_with_uncertainty(model_path: str, cache_path: Path, device: str):
    """Drop-in: pass llpr_cache_path to MACECalculator, read calc.results."""
    loaded = torch.load(model_path, map_location="cpu", weights_only=False)
    calc = MACECalculator(
        models=[loaded],
        device=device,
        default_dtype="float64",
        llpr_cache_path=str(cache_path),
        llpr_hook_layer="pre_last",
    )
    print(f"[infer] implemented_properties: {calc.implemented_properties}")

    # In-distribution frame
    atoms = bulk("Si", cubic=True, a=5.43, crystalstructure="diamond")
    atoms.rattle(stdev=0.03)
    atoms.calc = calc
    E = atoms.get_potential_energy()
    u = calc.results["uncertainty"]
    print(f"[infer] in-distribution Si8:")
    print(f"          E = {E:.4f} eV")
    print(f"          u_per_atom = {u}")
    print(f"          max_atom_uncertainty = {calc.results['max_atom_uncertainty']:.4g}")

    # Out-of-distribution frame (large rattle → unfamiliar geometry)
    atoms_ood = bulk("Si", cubic=True, a=5.43, crystalstructure="diamond")
    atoms_ood.rattle(stdev=0.30)
    atoms_ood.calc = calc
    E_ood = atoms_ood.get_potential_energy()
    u_ood = calc.results["uncertainty"]
    print(f"[infer] heavily-rattled Si8 (OOD):")
    print(f"          E = {E_ood:.4f} eV")
    print(f"          u_per_atom mean = {u_ood.mean():.4g}")
    print(f"          max_atom_uncertainty = {calc.results['max_atom_uncertainty']:.4g}")
    print(f"[infer] OOD u_max / in-dist u_max ratio: "
          f"{calc.results['max_atom_uncertainty'] / u.max():.2f}x")


def main():
    repo_root = Path(__file__).resolve().parent.parent
    model_path = (
        repo_root / "mace" / "calculators" / "foundations_models"
        / "2023-12-03-mace-mp.model"
    )
    if not model_path.exists():
        sys.exit(f"Bundled foundation model not found at {model_path}")

    device = "cpu"  # quickstart runs on CPU
    print(f"=== LLPR quickstart on {model_path.name} ({device}) ===\n")

    # 1. Build dataset
    print("[1/3] Building synthetic Si dataset (self-labeled by the foundation)")
    loaded = torch.load(str(model_path), map_location="cpu", weights_only=False)
    calc_ref = MACECalculator(
        models=[loaded], device="cpu", default_dtype="float64"
    )
    train, valid = build_synthetic_dataset(calc_ref, n_train=15, n_valid=8)
    print(f"        {len(train)} train + {len(valid)} valid frames\n")

    # 2. Build LLPR cache
    print("[2/3] Building LLPR cache (Cholesky-factored, Spearman-calibrated)")
    cache_path = build_cache(
        str(model_path), train, valid,
        device=device, hook_layer="pre_last",
    )
    print()

    # 3. Inference with uncertainty
    print("[3/3] Drop-in inference: MACECalculator(llpr_cache_path=...)")
    inference_with_uncertainty(str(model_path), cache_path, device=device)


if __name__ == "__main__":
    main()
