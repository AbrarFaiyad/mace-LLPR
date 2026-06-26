"""LLPR (Last-Layer Posterior Regression) uncertainty cache for MACE.

Implements per-atom Mahalanobis-distance uncertainty from Bigi, Chong, Ceriotti
& Grasselli, MLST 5(4) (2024), arXiv:2403.02251, eq. 24:

    sigma^2_star = alpha^2 * f_star^T (F^T F + sigma^2 * I)^{-1} f_star

where F is the training last-layer feature matrix, f_star is the new atom's
last-layer feature vector, sigma^2 is a regularizer chosen for numerical
stability of the Cholesky factorization, and alpha^2 is a calibration scale.

Numerics:
  - Cholesky factor L of (M = 0.5*(F^T F + (F^T F)^T) + sigma^2 * I) is stored
  - Per-atom uncertainty computed via triangular solve: u^2 = alpha^2 * ||L^{-T} phi^T||^2
  - fp64 throughout
  - Optional auto-PD search: smallest sigma^2 giving stable Cholesky
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import torch

try:
    from scipy.stats import spearmanr as _spearmanr
except ImportError:
    _spearmanr = None


CHECKPOINT_VERSION = 1


class LLPRCache:
    """Stores Cholesky factor of (Phi^T Phi + sigma^2 * I) plus calibration alpha^2.

    Workflow:
      1. Instantiate:    cache = LLPRCache(d=128, device='cuda')
      2. Accumulate:     for phi_batch in loader: cache.accumulate(phi_batch)
      3. (Optional)      cache.all_reduce()  if multi-rank
      4. Finalize:       cache.finalize(regularizer_sq=None)  # auto-PD
      5. (Optional)      cache.calibrate_spearman(...) or .calibrate_nll(...)
      6. Save:           cache.save('cache.pt')
      7. Load + use:     cache = LLPRCache.load('cache.pt', device='cuda')
                         u = cache.u_per_atom(phi)
    """

    def __init__(
        self,
        d: int,
        device: Optional[torch.device | str] = None,
        dtype: torch.dtype = torch.float64,
    ):
        if d <= 0:
            raise ValueError(f"d must be positive, got {d}")
        self.d = int(d)
        self.device = torch.device(device) if device is not None else torch.device("cpu")
        self.dtype = dtype
        # accumulator (used during cache build)
        self._covariance: Optional[torch.Tensor] = torch.zeros(
            d, d, dtype=dtype, device=self.device
        )
        # populated after finalize()
        self.cholesky: Optional[torch.Tensor] = None
        self.regularizer_sq: float = 0.0
        self.alpha_sq: float = 1.0

    # ------------------------------------------------------------------
    # Build path
    # ------------------------------------------------------------------
    @torch.no_grad()
    def accumulate(self, phi_batch: torch.Tensor) -> None:
        """Add `phi_batch.T @ phi_batch` to the running covariance.

        phi_batch: [N_atoms_in_batch, d] tensor. Will be cast to fp64.
        """
        if self._covariance is None:
            raise RuntimeError(
                "Cannot accumulate after finalize(); reset by building a new LLPRCache"
            )
        if phi_batch.ndim != 2 or phi_batch.shape[1] != self.d:
            raise ValueError(
                f"phi_batch shape {tuple(phi_batch.shape)} != [N, {self.d}]"
            )
        phi = phi_batch.to(device=self.device, dtype=self.dtype)
        self._covariance += phi.T @ phi

    def all_reduce(self) -> None:
        """Sum covariance accumulators across distributed ranks (no-op if not initialized)."""
        if (
            torch.distributed.is_available()
            and torch.distributed.is_initialized()
            and self._covariance is not None
        ):
            torch.distributed.all_reduce(
                self._covariance, op=torch.distributed.ReduceOp.SUM
            )

    def finalize(self, regularizer_sq: Optional[float] = None) -> None:
        """Symmetrize covariance, add sigma^2 * I, compute Cholesky factor.

        regularizer_sq=None: auto-search smallest sigma^2 making M positive-definite.
        regularizer_sq=float: use that value directly.
        """
        if self._covariance is None:
            raise RuntimeError("finalize() called twice or no accumulate() called")
        sym = 0.5 * (self._covariance + self._covariance.T)
        eye = torch.eye(self.d, dtype=self.dtype, device=self.device)

        if regularizer_sq is not None:
            if regularizer_sq < 0:
                raise ValueError(f"regularizer_sq must be >= 0, got {regularizer_sq}")
            try:
                self.cholesky = torch.linalg.cholesky(sym + float(regularizer_sq) * eye)
                self.regularizer_sq = float(regularizer_sq)
            except RuntimeError as exc:
                raise RuntimeError(
                    f"Cholesky failed at requested regularizer_sq={regularizer_sq:.4g}; "
                    "try a larger value or use auto-PD (regularizer_sq=None)"
                ) from exc
        else:
            r = 1e-20
            success = False
            while r < 1e16:
                try:
                    self.cholesky = torch.linalg.cholesky(sym + r * eye)
                    self.regularizer_sq = float(r)
                    success = True
                    break
                except RuntimeError:
                    r *= 10.0
            if not success:
                raise RuntimeError(
                    "Auto-PD Cholesky failed at sigma^2=1e16; inspect feature matrix"
                )

        # release accumulator memory after factorization
        self._covariance = None

    # ------------------------------------------------------------------
    # Inference path
    # ------------------------------------------------------------------
    def u_per_atom(self, phi: torch.Tensor) -> torch.Tensor:
        """Compute per-atom uncertainty u_i for atoms with features phi [N, d].

        u_i = sqrt( alpha^2 * phi_i^T * M^{-1} * phi_i )
            = sqrt( alpha^2 * || L^{-T} phi_i^T ||^2 )

        Implemented via triangular solve (no explicit inverse).
        Returns [N] tensor in fp64 on the cache device. Differentiable wrt phi
        if caller invoked the model forward with grad enabled.
        """
        if self.cholesky is None:
            raise RuntimeError("LLPRCache not finalized; call finalize() first")
        if phi.ndim != 2 or phi.shape[1] != self.d:
            raise ValueError(f"phi shape {tuple(phi.shape)} != [N, {self.d}]")
        phi_d = phi.to(device=self.device, dtype=self.dtype)
        v = torch.linalg.solve_triangular(self.cholesky, phi_d.T, upper=False)
        u_sq = self.alpha_sq * (v * v).sum(dim=0)
        return u_sq.clamp_min(1e-12).sqrt()

    # ------------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------------
    def calibrate_spearman(
        self,
        phi_valid: torch.Tensor,
        F_err_valid: torch.Tensor,
        lambdas_sq: Iterable[float] = (1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0),
    ) -> Tuple[float, float, List[Tuple[float, float]]]:
        """Sweep regularizer_sq, pick value maximizing Spearman(u, F_err) on valid set.

        Rebuilds the Cholesky factor with the selected regularizer at the end.
        Returns (best_lambda_sq, best_spearman, all_results).
        alpha_sq remains at its current value (default 1.0).
        """
        if _spearmanr is None:
            raise ImportError("scipy.stats.spearmanr required for calibrate_spearman")
        if phi_valid.shape[0] != F_err_valid.shape[0]:
            raise ValueError(
                f"phi_valid N={phi_valid.shape[0]} != F_err N={F_err_valid.shape[0]}"
            )
        if phi_valid.shape[1] != self.d:
            raise ValueError(f"phi_valid d={phi_valid.shape[1]} != cache d={self.d}")
        if self._covariance is None and self.cholesky is None:
            raise RuntimeError("call accumulate() before calibrate_spearman")

        sym = (
            0.5 * (self._covariance + self._covariance.T)
            if self._covariance is not None
            else self._covariance_from_cholesky()
        )
        eye = torch.eye(self.d, dtype=self.dtype, device=self.device)
        phi_v_d = phi_valid.to(device=self.device, dtype=self.dtype)
        F_err_np = F_err_valid.detach().cpu().numpy()

        results: List[Tuple[float, float]] = []
        for lam_sq in lambdas_sq:
            try:
                L = torch.linalg.cholesky(sym + float(lam_sq) * eye)
                v = torch.linalg.solve_triangular(L, phi_v_d.T, upper=False)
                u_sq = (v * v).sum(dim=0)
                u = u_sq.clamp_min(1e-12).sqrt().detach().cpu().numpy()
                rho, _ = _spearmanr(u, F_err_np)
                if rho is None or rho != rho:  # NaN guard
                    rho = -1.0
            except Exception as exc:
                print(f"[LLPR calibrate_spearman] lambda_sq={lam_sq:.4g}: {exc}")
                rho = -1.0
            results.append((float(lam_sq), float(rho)))

        best = max(results, key=lambda r: r[1])
        # rebuild cholesky at the chosen regularizer
        self._covariance = sym  # restore so finalize works
        self.cholesky = None
        self.finalize(regularizer_sq=best[0])
        return best[0], best[1], results

    def calibrate_nll(
        self,
        phi_valid: torch.Tensor,
        residuals: torch.Tensor,
    ) -> float:
        """Closed-form alpha^2 from squared residuals (metatrain-style NLL calibration).

        alpha^2 = mean(residuals^2) / mean(uncalibrated_u^2)

        residuals: [N_valid_targets] (e.g., per-frame energy errors). Must match
        granularity of uncalibrated u^2 computed from phi_valid via per-atom sum
        or per-system feature, depending on how phi_valid was assembled.
        """
        if self.cholesky is None:
            raise RuntimeError("call finalize() before calibrate_nll")
        if phi_valid.shape[1] != self.d:
            raise ValueError(f"phi_valid d={phi_valid.shape[1]} != cache d={self.d}")
        phi_v_d = phi_valid.to(device=self.device, dtype=self.dtype)
        v = torch.linalg.solve_triangular(self.cholesky, phi_v_d.T, upper=False)
        u_sq_uncalib = (v * v).sum(dim=0).detach().cpu()
        res_sq = residuals.to(torch.float64).detach().cpu() ** 2
        if u_sq_uncalib.shape != res_sq.shape:
            raise ValueError(
                f"u_sq shape {u_sq_uncalib.shape} != residuals^2 shape {res_sq.shape}"
            )
        mean_u_sq = float(u_sq_uncalib.mean())
        if mean_u_sq <= 0:
            raise RuntimeError("mean uncalibrated u^2 is non-positive; cannot calibrate")
        self.alpha_sq = float(res_sq.mean()) / mean_u_sq
        return self.alpha_sq

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(self, path: Path | str) -> None:
        if self.cholesky is None:
            raise RuntimeError("Cannot save before finalize()")
        torch.save(
            {
                "cholesky": self.cholesky.detach().cpu(),
                "regularizer_sq": float(self.regularizer_sq),
                "alpha_sq": float(self.alpha_sq),
                "d": int(self.d),
                "version": CHECKPOINT_VERSION,
            },
            str(path),
        )

    @classmethod
    def load(cls, path: Path | str, device: Optional[torch.device | str] = None) -> "LLPRCache":
        cache_dict = torch.load(str(path), map_location="cpu", weights_only=False)
        if int(cache_dict.get("version", 0)) > CHECKPOINT_VERSION:
            raise RuntimeError(
                f"LLPR cache version {cache_dict['version']} newer than supported "
                f"{CHECKPOINT_VERSION}; upgrade mace"
            )
        d = int(cache_dict["d"])
        chol = cache_dict["cholesky"]
        target_device = torch.device(device) if device is not None else torch.device("cpu")
        instance = cls(d=d, device=target_device, dtype=chol.dtype)
        instance._covariance = None  # loaded cache is finalized
        instance.cholesky = chol.to(device=target_device)
        instance.regularizer_sq = float(cache_dict["regularizer_sq"])
        instance.alpha_sq = float(cache_dict["alpha_sq"])
        return instance

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _covariance_from_cholesky(self) -> torch.Tensor:
        """Reconstruct (FtF + sigma^2 I) from stored Cholesky factor; subtract sigma^2 I."""
        if self.cholesky is None:
            raise RuntimeError("no cholesky stored")
        M = self.cholesky @ self.cholesky.T
        eye = torch.eye(self.d, dtype=self.dtype, device=self.device)
        return M - self.regularizer_sq * eye

    def __repr__(self) -> str:
        finalized = "finalized" if self.cholesky is not None else "accumulating"
        return (
            f"LLPRCache(d={self.d}, {finalized}, "
            f"reg_sq={self.regularizer_sq:.4g}, alpha_sq={self.alpha_sq:.4g})"
        )
