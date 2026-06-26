"""Unit tests for mace.modules.llpr.LLPRCache and mace.tools.llpr_features.

No MACE-specific dependencies — uses a fake model that mimics the
`readouts[-1].linear` attribute path so the forward-pre-hook fires.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn

from mace.modules.llpr import LLPRCache
from mace.tools.llpr_features import (
    LastLayerFeatureExtractor,
    VALID_HOOK_LAYERS,
    _infer_in_dim,
)


# -----------------------------------------------------------------------------
# Fakes
# -----------------------------------------------------------------------------
class _FakeLinearReadout(nn.Module):
    """Mimics MACE LinearReadoutBlock (single `linear` attribute)."""

    def __init__(self, d: int):
        super().__init__()
        self.linear = nn.Linear(d, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class _FakeNonLinearReadout(nn.Module):
    """Mimics MACE NonLinearReadoutBlock (linear_1 -> nonlinearity -> linear_2)."""

    def __init__(self, d_in: int, d_hidden: int = 16):
        super().__init__()
        self.linear_1 = nn.Linear(d_in, d_hidden, bias=False)
        self.nonlinearity = nn.SiLU()
        self.linear_2 = nn.Linear(d_hidden, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear_2(self.nonlinearity(self.linear_1(x)))


class _FakeMACE(nn.Module):
    """Minimal stand-in: routes input through readouts[-1] so the hook fires."""

    def __init__(self, readout: nn.Module):
        super().__init__()
        self.readouts = nn.ModuleList([readout])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.readouts[-1](x)


# -----------------------------------------------------------------------------
# LLPRCache: construction + accumulation
# -----------------------------------------------------------------------------
def test_cache_init_rejects_bad_d():
    with pytest.raises(ValueError):
        LLPRCache(d=0)
    with pytest.raises(ValueError):
        LLPRCache(d=-1)


def test_accumulate_rejects_wrong_shape():
    cache = LLPRCache(d=8)
    with pytest.raises(ValueError):
        cache.accumulate(torch.zeros(5))  # 1D
    with pytest.raises(ValueError):
        cache.accumulate(torch.zeros(10, 9))  # wrong d


def test_accumulate_then_finalize():
    cache = LLPRCache(d=12)
    phi = torch.randn(300, 12, dtype=torch.float64)
    cache.accumulate(phi)
    cache.finalize(regularizer_sq=1.0)
    assert cache.cholesky is not None
    assert tuple(cache.cholesky.shape) == (12, 12)
    assert cache.regularizer_sq == 1.0
    assert cache._covariance is None  # released after finalize


def test_finalize_rejects_negative_regularizer():
    cache = LLPRCache(d=6)
    cache.accumulate(torch.randn(100, 6, dtype=torch.float64))
    with pytest.raises(ValueError):
        cache.finalize(regularizer_sq=-0.1)


def test_finalize_double_call_raises():
    cache = LLPRCache(d=6)
    cache.accumulate(torch.randn(100, 6, dtype=torch.float64))
    cache.finalize(regularizer_sq=1.0)
    with pytest.raises(RuntimeError):
        cache.finalize(regularizer_sq=1.0)


def test_accumulate_after_finalize_raises():
    cache = LLPRCache(d=6)
    cache.accumulate(torch.randn(100, 6, dtype=torch.float64))
    cache.finalize(regularizer_sq=1.0)
    with pytest.raises(RuntimeError):
        cache.accumulate(torch.randn(10, 6, dtype=torch.float64))


# -----------------------------------------------------------------------------
# Cholesky correctness
# -----------------------------------------------------------------------------
def test_cholesky_is_lower_triangular():
    cache = LLPRCache(d=10)
    cache.accumulate(torch.randn(500, 10, dtype=torch.float64))
    cache.finalize(regularizer_sq=1.0)
    upper = torch.triu(cache.cholesky, diagonal=1)
    assert torch.allclose(upper, torch.zeros_like(upper), atol=1e-12)


def test_cholesky_factors_M_correctly():
    """Verify L @ L.T == 0.5*(PtP + PtP.T) + reg*I."""
    torch.manual_seed(0)
    d = 8
    cache = LLPRCache(d=d)
    phi = torch.randn(400, d, dtype=torch.float64)
    cache.accumulate(phi)
    reg = 0.5
    sym_PtP = 0.5 * (phi.T @ phi + (phi.T @ phi).T)
    M_expected = sym_PtP + reg * torch.eye(d, dtype=torch.float64)
    cache.finalize(regularizer_sq=reg)
    M_reconstructed = cache.cholesky @ cache.cholesky.T
    assert torch.allclose(M_reconstructed, M_expected, atol=1e-10)


def test_symmetrization_handles_asymmetric_accumulator():
    """Manually inject asymmetric noise; symmetrization should still produce PD M."""
    cache = LLPRCache(d=6)
    cache.accumulate(torch.randn(200, 6, dtype=torch.float64))
    # asymmetric perturbation
    noise = torch.randn(6, 6, dtype=torch.float64) * 1e-8
    cache._covariance += noise  # not symmetric
    cache.finalize(regularizer_sq=1.0)
    # should succeed; check Cholesky is real and lower-triangular
    assert cache.cholesky is not None
    assert torch.isreal(cache.cholesky).all()


# -----------------------------------------------------------------------------
# Auto-PD search
# -----------------------------------------------------------------------------
def test_auto_pd_finds_regularizer_on_rank_deficient_features():
    """Build a rank-deficient training matrix; auto-PD should still succeed."""
    torch.manual_seed(7)
    d = 16
    # Only 4 independent directions, rest are zeros
    n = 200
    base = torch.randn(n, 4, dtype=torch.float64)
    phi = torch.cat([base, torch.zeros(n, d - 4, dtype=torch.float64)], dim=1)
    cache = LLPRCache(d=d)
    cache.accumulate(phi)
    cache.finalize(regularizer_sq=None)  # auto
    assert cache.cholesky is not None
    assert cache.regularizer_sq > 0
    # Should pick something larger than 1e-20 (the start of the search)
    assert cache.regularizer_sq >= 1e-20


def test_auto_pd_uses_small_regularizer_when_features_well_conditioned():
    """Well-conditioned random features: auto-PD picks small regularizer."""
    torch.manual_seed(0)
    d = 8
    cache = LLPRCache(d=d)
    cache.accumulate(torch.randn(2000, d, dtype=torch.float64))
    cache.finalize(regularizer_sq=None)
    # 2000 >> 8 → covariance is full-rank with eigenvalues O(N) >> 1
    # auto-PD should pick something <= 1e-10
    assert cache.regularizer_sq <= 1e-10


# -----------------------------------------------------------------------------
# Equivalence: Cholesky path vs full inverse path
# -----------------------------------------------------------------------------
def test_u_per_atom_matches_full_inverse_formula():
    """Verify u_i computed via triangular solve == sqrt(alpha^2 * phi^T M_inv phi)."""
    torch.manual_seed(42)
    d = 12
    n_train, n_test = 500, 30
    cache = LLPRCache(d=d)
    phi_train = torch.randn(n_train, d, dtype=torch.float64)
    cache.accumulate(phi_train)
    cache.finalize(regularizer_sq=0.5)
    cache.alpha_sq = 2.0

    phi_test = torch.randn(n_test, d, dtype=torch.float64)

    # Cholesky path
    u_chol = cache.u_per_atom(phi_test)

    # Reference: full inverse path
    sym_PtP = 0.5 * (phi_train.T @ phi_train + (phi_train.T @ phi_train).T)
    M = sym_PtP + 0.5 * torch.eye(d, dtype=torch.float64)
    M_inv = torch.linalg.inv(M)
    u_sq_ref = 2.0 * (phi_test @ M_inv * phi_test).sum(dim=-1)
    u_ref = u_sq_ref.clamp_min(1e-12).sqrt()

    assert torch.allclose(u_chol, u_ref, atol=1e-9)


# -----------------------------------------------------------------------------
# u_per_atom: shapes, OOD detection, autograd
# -----------------------------------------------------------------------------
def test_u_per_atom_rejects_wrong_shape():
    cache = LLPRCache(d=8)
    cache.accumulate(torch.randn(100, 8, dtype=torch.float64))
    cache.finalize(regularizer_sq=1.0)
    with pytest.raises(ValueError):
        cache.u_per_atom(torch.randn(5, 9))


def test_u_per_atom_before_finalize_raises():
    cache = LLPRCache(d=8)
    with pytest.raises(RuntimeError):
        cache.u_per_atom(torch.randn(5, 8))


def test_u_increases_for_OOD():
    """LLPR core property: features far from training cloud -> higher u."""
    torch.manual_seed(7)
    d = 16
    cache = LLPRCache(d=d)
    # tight training cloud
    cache.accumulate(torch.randn(3000, d, dtype=torch.float64) * 0.5)
    cache.finalize(regularizer_sq=0.01)

    # in-distribution
    x_in = torch.randn(200, d, dtype=torch.float64) * 0.5
    u_in = cache.u_per_atom(x_in).mean().item()

    # OOD (10x farther from origin)
    x_out = torch.randn(200, d, dtype=torch.float64) * 5.0
    u_out = cache.u_per_atom(x_out).mean().item()

    assert u_out > u_in * 3, f"u_out={u_out:.3g} not >> u_in={u_in:.3g}"


def test_u_differentiable_through_phi():
    """grad flows: u should be differentiable wrt phi when caller enables grad."""
    cache = LLPRCache(d=8)
    cache.accumulate(torch.randn(300, 8, dtype=torch.float64))
    cache.finalize(regularizer_sq=1.0)

    phi = torch.randn(5, 8, dtype=torch.float64, requires_grad=True)
    u = cache.u_per_atom(phi)
    loss = u.sum()
    loss.backward()
    assert phi.grad is not None
    assert phi.grad.shape == phi.shape
    assert torch.all(torch.isfinite(phi.grad))


# -----------------------------------------------------------------------------
# Calibration
# -----------------------------------------------------------------------------
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_calibrate_spearman_recovers_correlated_signal(seed):
    """Build synthetic Φ where OOD atoms have larger F_err; sweep λ² and verify
    best Spearman > 0.5."""
    rng = np.random.RandomState(seed)
    torch.manual_seed(seed)
    d = 8

    # training cloud anisotropic (heavy in some directions, thin in others)
    scales = torch.tensor([10.0, 0.1, 5.0, 0.1, 1.0, 0.5, 2.0, 0.1], dtype=torch.float64)
    n_train = 1000
    phi_train = torch.randn(n_train, d, dtype=torch.float64) * scales

    cache = LLPRCache(d=d)
    cache.accumulate(phi_train)

    # valid set: half in-dist, half OOD along thin axes
    n_valid = 100
    phi_valid = torch.randn(n_valid, d, dtype=torch.float64) * scales
    phi_valid[n_valid // 2:] += torch.tensor(
        [0.0, 5.0, 0.0, 5.0, 0.0, 0.0, 0.0, 5.0], dtype=torch.float64
    )
    F_err = torch.cat([
        torch.from_numpy(rng.rand(n_valid // 2) * 0.1),
        1.0 + torch.from_numpy(rng.rand(n_valid - n_valid // 2) * 0.5),
    ]).to(torch.float64)

    best_lam, best_rho, results = cache.calibrate_spearman(phi_valid, F_err)
    assert best_rho > 0.5, f"expected Spearman > 0.5, got {best_rho:.3f}"
    assert best_lam in [r[0] for r in results]


def test_calibrate_spearman_rebuilds_cholesky_at_best_lambda():
    torch.manual_seed(0)
    d = 6
    cache = LLPRCache(d=d)
    cache.accumulate(torch.randn(300, d, dtype=torch.float64))
    phi_valid = torch.randn(50, d, dtype=torch.float64)
    F_err = torch.from_numpy(np.random.rand(50)).to(torch.float64)
    best_lam, _, _ = cache.calibrate_spearman(phi_valid, F_err, lambdas_sq=[1e-3, 1.0])
    assert cache.cholesky is not None
    assert cache.regularizer_sq == best_lam


def test_calibrate_nll_returns_positive_alpha():
    """alpha^2 = mean(res^2) / mean(uncal_u^2); should be positive."""
    torch.manual_seed(0)
    d = 8
    cache = LLPRCache(d=d)
    cache.accumulate(torch.randn(500, d, dtype=torch.float64))
    cache.finalize(regularizer_sq=1.0)
    phi_valid = torch.randn(40, d, dtype=torch.float64)
    residuals = torch.randn(40, dtype=torch.float64)
    alpha_sq = cache.calibrate_nll(phi_valid, residuals)
    assert alpha_sq > 0
    assert cache.alpha_sq == alpha_sq


# -----------------------------------------------------------------------------
# Save / load
# -----------------------------------------------------------------------------
def test_save_load_roundtrip(tmp_path):
    cache = LLPRCache(d=10)
    cache.accumulate(torch.randn(200, 10, dtype=torch.float64))
    cache.finalize(regularizer_sq=0.7)
    cache.alpha_sq = 1.5

    p = tmp_path / "cache.pt"
    cache.save(p)
    loaded = LLPRCache.load(p)
    assert loaded.d == cache.d
    assert loaded.regularizer_sq == 0.7
    assert loaded.alpha_sq == 1.5
    assert torch.allclose(loaded.cholesky, cache.cholesky.cpu(), atol=1e-12)


def test_save_before_finalize_raises(tmp_path):
    cache = LLPRCache(d=8)
    with pytest.raises(RuntimeError):
        cache.save(tmp_path / "bad.pt")


def test_load_predictions_match_original(tmp_path):
    """End-to-end: build, save, load, verify u_per_atom is bit-identical."""
    torch.manual_seed(0)
    d = 12
    cache = LLPRCache(d=d)
    cache.accumulate(torch.randn(400, d, dtype=torch.float64))
    cache.finalize(regularizer_sq=0.3)
    cache.alpha_sq = 0.9

    p = tmp_path / "cache.pt"
    cache.save(p)
    loaded = LLPRCache.load(p)

    phi_test = torch.randn(15, d, dtype=torch.float64)
    u_orig = cache.u_per_atom(phi_test)
    u_load = loaded.u_per_atom(phi_test)
    assert torch.allclose(u_orig, u_load, atol=1e-12)


# -----------------------------------------------------------------------------
# LastLayerFeatureExtractor
# -----------------------------------------------------------------------------
def test_extractor_linear_readout_auto_and_pre_last_equivalent():
    """For LinearReadoutBlock both hook_layer modes target `linear`."""
    d = 8
    model = _FakeMACE(_FakeLinearReadout(d))
    e_auto = LastLayerFeatureExtractor(model, hook_layer="auto")
    e_pre = LastLayerFeatureExtractor(model, hook_layer="pre_last")
    assert e_auto.d == e_pre.d == d
    e_auto.remove()
    e_pre.remove()


def test_extractor_nonlinear_readout_dims():
    """NonLinearReadoutBlock: auto -> linear_2 input (d_hidden), pre_last -> linear_1 input (d_in)."""
    d_in, d_hidden = 64, 8
    model = _FakeMACE(_FakeNonLinearReadout(d_in, d_hidden))
    e_auto = LastLayerFeatureExtractor(model, hook_layer="auto")
    e_pre = LastLayerFeatureExtractor(model, hook_layer="pre_last")
    assert e_auto.d == d_hidden
    assert e_pre.d == d_in
    e_auto.remove()
    e_pre.remove()


def test_extractor_captures_correct_shape():
    d_in = 32
    model = _FakeMACE(_FakeNonLinearReadout(d_in, d_hidden=8))
    extractor = LastLayerFeatureExtractor(model, hook_layer="pre_last")
    x = torch.randn(7, d_in)
    with torch.no_grad():
        model(x)
    phi = extractor.last_captured()
    assert phi.shape == (7, d_in)
    extractor.remove()


def test_extractor_reset_clears_capture():
    model = _FakeMACE(_FakeLinearReadout(8))
    extractor = LastLayerFeatureExtractor(model, hook_layer="auto")
    with torch.no_grad():
        model(torch.randn(3, 8))
    assert extractor._captured is not None
    extractor.reset()
    assert extractor._captured is None
    with pytest.raises(RuntimeError):
        extractor.last_captured()
    extractor.remove()


def test_extractor_invalid_hook_layer_raises():
    model = _FakeMACE(_FakeLinearReadout(8))
    with pytest.raises(ValueError):
        LastLayerFeatureExtractor(model, hook_layer="bogus")


def test_extractor_no_readouts_raises():
    class Empty(nn.Module):
        pass
    with pytest.raises(RuntimeError):
        LastLayerFeatureExtractor(Empty(), hook_layer="auto")


def test_extractor_context_manager_removes_hook():
    model = _FakeMACE(_FakeLinearReadout(8))
    with LastLayerFeatureExtractor(model, hook_layer="auto") as extractor:
        assert extractor._handle is not None
        with torch.no_grad():
            model(torch.randn(2, 8))
        assert extractor.last_captured().shape == (2, 8)
    # after __exit__, handle should be removed
    assert extractor._handle is None


# -----------------------------------------------------------------------------
# _infer_in_dim helper
# -----------------------------------------------------------------------------
def test_infer_in_dim_torch_linear():
    layer = nn.Linear(64, 16)
    assert _infer_in_dim(layer) == 64


def test_infer_in_dim_no_attribute_raises():
    class Bogus(nn.Module):
        pass
    with pytest.raises(RuntimeError):
        _infer_in_dim(Bogus())


# -----------------------------------------------------------------------------
# End-to-end with extractor + LLPRCache
# -----------------------------------------------------------------------------
def test_e2e_extractor_to_cache_to_u_per_atom():
    """Wire LastLayerFeatureExtractor into LLPRCache build + inference path."""
    torch.manual_seed(0)
    d_in = 32
    model = _FakeMACE(_FakeNonLinearReadout(d_in, d_hidden=8))
    extractor = LastLayerFeatureExtractor(model, hook_layer="pre_last")
    assert extractor.d == d_in

    cache = LLPRCache(d=d_in)
    # accumulate from "training" forward passes
    for _ in range(20):
        x = torch.randn(50, d_in)
        with torch.no_grad():
            model(x)
        cache.accumulate(extractor.last_captured().double())

    cache.finalize(regularizer_sq=None)
    assert cache.regularizer_sq > 0

    # inference on new x
    x_test = torch.randn(10, d_in)
    with torch.no_grad():
        model(x_test)
    u = cache.u_per_atom(extractor.last_captured())
    assert u.shape == (10,)
    assert torch.all(u >= 0)
    assert torch.all(torch.isfinite(u))

    extractor.remove()


# -----------------------------------------------------------------------------
# Session 3: MACECalculator integration tests
# These require the bundled MACE-MP foundation model.
# -----------------------------------------------------------------------------
import os as _os
from pathlib import Path as _Path

_BUNDLED_MP = (
    _Path(__file__).parent.parent
    / "mace"
    / "calculators"
    / "foundations_models"
    / "2023-12-03-mace-mp.model"
)

_HAS_MP = _BUNDLED_MP.exists()


@pytest.mark.skipif(not _HAS_MP, reason="bundled MACE-MP foundation not present")
def test_mace_calculator_without_llpr_is_unchanged():
    """Backward-compat: no llpr_cache_path -> no uncertainty in results."""
    from ase.build import bulk
    from mace.calculators.mace import MACECalculator

    loaded = torch.load(str(_BUNDLED_MP), map_location="cpu", weights_only=False)
    calc = MACECalculator(models=[loaded], device="cpu", default_dtype="float64")
    atoms = bulk("Si", cubic=True, a=5.43, crystalstructure="diamond")
    atoms.calc = calc
    atoms.get_potential_energy()
    assert "uncertainty" not in calc.results
    assert "max_atom_uncertainty" not in calc.results
    assert "uncertainty" not in calc.implemented_properties
    assert calc._llpr_cache is None
    assert calc._llpr_extractor is None


@pytest.mark.skipif(not _HAS_MP, reason="bundled MACE-MP foundation not present")
def test_mace_calculator_with_llpr_returns_uncertainty(tmp_path):
    """End-to-end: build cache from in-memory Si frames, load, verify u finite."""
    from ase.build import bulk
    from mace.calculators.mace import MACECalculator
    from mace.modules.llpr import LLPRCache
    from mace.tools.llpr_features import LastLayerFeatureExtractor

    loaded = torch.load(str(_BUNDLED_MP), map_location="cpu", weights_only=False)

    # build cache manually
    calc_for_build = MACECalculator(
        models=[loaded], device="cpu", default_dtype="float64"
    )
    extractor = LastLayerFeatureExtractor(
        calc_for_build.models[0], hook_layer="pre_last"
    )
    cache = LLPRCache(d=extractor.d, device="cpu")
    rng = np.random.RandomState(0)
    for _ in range(10):
        atoms = bulk("Si", cubic=True, a=5.43 + 0.05 * rng.randn(),
                     crystalstructure="diamond")
        atoms.rattle(stdev=0.05, seed=rng.randint(1 << 30))
        batch = calc_for_build._atoms_to_batch(atoms).to_dict()
        with torch.no_grad():
            try:
                calc_for_build.models[0](batch, compute_force=False)
            except TypeError:
                calc_for_build.models[0](batch)
        cache.accumulate(extractor.last_captured().double())
    cache.finalize(regularizer_sq=None)
    cache_path = tmp_path / "test_cache.pt"
    cache.save(cache_path)
    extractor.remove()

    # load fresh model + use cache via the calculator kwarg
    loaded2 = torch.load(str(_BUNDLED_MP), map_location="cpu", weights_only=False)
    calc = MACECalculator(
        models=[loaded2],
        device="cpu",
        default_dtype="float64",
        llpr_cache_path=str(cache_path),
        llpr_hook_layer="pre_last",
    )
    assert "uncertainty" in calc.implemented_properties
    assert "max_atom_uncertainty" in calc.implemented_properties
    assert calc._llpr_cache is not None
    assert calc._llpr_extractor is not None
    assert calc._llpr_extractor.d == cache.d

    atoms = bulk("Si", cubic=True, a=5.43, crystalstructure="diamond")
    atoms.rattle(stdev=0.1)
    atoms.calc = calc
    atoms.get_potential_energy()
    u = calc.results["uncertainty"]
    assert u.shape == (len(atoms),)
    assert np.all(np.isfinite(u))
    assert np.all(u >= 0)
    assert calc.results["max_atom_uncertainty"] == float(u.max())


@pytest.mark.skipif(not _HAS_MP, reason="bundled MACE-MP foundation not present")
def test_mace_calculator_llpr_matches_manual_path(tmp_path):
    """The built-in calculator path must equal the manual hook+cache path."""
    from ase.build import bulk
    from mace.calculators.mace import MACECalculator
    from mace.modules.llpr import LLPRCache
    from mace.tools.llpr_features import LastLayerFeatureExtractor

    loaded = torch.load(str(_BUNDLED_MP), map_location="cpu", weights_only=False)

    # build cache
    calc_for_build = MACECalculator(
        models=[loaded], device="cpu", default_dtype="float64"
    )
    extractor = LastLayerFeatureExtractor(
        calc_for_build.models[0], hook_layer="pre_last"
    )
    cache = LLPRCache(d=extractor.d, device="cpu")
    for _ in range(8):
        atoms = bulk("Si", cubic=True, a=5.43, crystalstructure="diamond")
        atoms.rattle(stdev=0.05)
        batch = calc_for_build._atoms_to_batch(atoms).to_dict()
        with torch.no_grad():
            try:
                calc_for_build.models[0](batch, compute_force=False)
            except TypeError:
                calc_for_build.models[0](batch)
        cache.accumulate(extractor.last_captured().double())
    cache.finalize(regularizer_sq=1.0)
    cache_path = tmp_path / "match_cache.pt"
    cache.save(cache_path)
    extractor.remove()

    atoms = bulk("Si", cubic=True, a=5.43, crystalstructure="diamond")
    atoms.rattle(stdev=0.15)

    # Path A: built-in calculator
    loaded_a = torch.load(str(_BUNDLED_MP), map_location="cpu", weights_only=False)
    calc_a = MACECalculator(
        models=[loaded_a],
        device="cpu",
        default_dtype="float64",
        llpr_cache_path=str(cache_path),
    )
    atoms_a = atoms.copy()
    atoms_a.calc = calc_a
    atoms_a.get_potential_energy()
    u_a = calc_a.results["uncertainty"]

    # Path B: manual
    loaded_b = torch.load(str(_BUNDLED_MP), map_location="cpu", weights_only=False)
    calc_b = MACECalculator(models=[loaded_b], device="cpu", default_dtype="float64")
    extr_b = LastLayerFeatureExtractor(calc_b.models[0], hook_layer="pre_last")
    cache_b = LLPRCache.load(cache_path, device="cpu")
    atoms_b = atoms.copy()
    batch = calc_b._atoms_to_batch(atoms_b).to_dict()
    with torch.no_grad():
        try:
            calc_b.models[0](batch, compute_force=False)
        except TypeError:
            calc_b.models[0](batch)
    u_b = cache_b.u_per_atom(extr_b.last_captured()).numpy()
    extr_b.remove()

    assert np.allclose(u_a, u_b, atol=1e-10), (
        f"built-in {u_a} != manual {u_b} (diff {np.abs(u_a - u_b).max():.4e})"
    )


@pytest.mark.skipif(not _HAS_MP, reason="bundled MACE-MP foundation not present")
def test_mace_calculator_llpr_committee_raises():
    """Committee mode + LLPR must raise NotImplementedError."""
    from mace.calculators.mace import MACECalculator

    loaded1 = torch.load(str(_BUNDLED_MP), map_location="cpu", weights_only=False)
    loaded2 = torch.load(str(_BUNDLED_MP), map_location="cpu", weights_only=False)
    with pytest.raises(NotImplementedError, match="committee"):
        MACECalculator(
            models=[loaded1, loaded2],
            device="cpu",
            default_dtype="float64",
            llpr_cache_path="/nonexistent_path.pt",  # never reached
        )


@pytest.mark.skipif(not _HAS_MP, reason="bundled MACE-MP foundation not present")
def test_mace_calculator_llpr_dim_mismatch_raises(tmp_path):
    """Cache with wrong d should be rejected on construction."""
    from mace.calculators.mace import MACECalculator
    from mace.modules.llpr import LLPRCache

    # Build a cache with d=999 (definitely doesn't match MACE-MP d=128)
    fake_cache = LLPRCache(d=999, device="cpu")
    fake_cache.accumulate(torch.randn(50, 999, dtype=torch.float64))
    fake_cache.finalize(regularizer_sq=1.0)
    cache_path = tmp_path / "wrong_d.pt"
    fake_cache.save(cache_path)

    loaded = torch.load(str(_BUNDLED_MP), map_location="cpu", weights_only=False)
    with pytest.raises(ValueError, match="does not match"):
        MACECalculator(
            models=[loaded],
            device="cpu",
            default_dtype="float64",
            llpr_cache_path=str(cache_path),
        )
