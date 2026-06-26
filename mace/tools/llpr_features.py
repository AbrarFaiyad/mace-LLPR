"""Forward-pre-hook helper to capture last-layer features from any MACE model.

Used by the LLPR uncertainty pipeline to extract `phi_i` (the per-atom feature
vector entering the last readout linear layer). Supports both MACE's
LinearReadoutBlock (single `linear` attribute) and NonLinearReadoutBlock
(`linear_1` -> nonlinearity -> `linear_2`).

Typical usage:

    from mace.tools.llpr_features import LastLayerFeatureExtractor

    extractor = LastLayerFeatureExtractor(model, hook_layer="pre_last")
    with torch.no_grad():
        model(batch_dict, compute_force=False)
    phi = extractor.last_captured()    # [N_atoms_in_batch, d]
    extractor.remove()                  # clean up the hook

Or as a context manager:

    with LastLayerFeatureExtractor(model, hook_layer="pre_last") as extractor:
        model(batch_dict)
        phi = extractor.last_captured()
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


VALID_HOOK_LAYERS = ("auto", "pre_last")


def _infer_in_dim(module: nn.Module) -> int:
    """Resolve the input feature dim entering `module`.

    Handles:
      - torch.nn.Linear (has `in_features`)
      - e3nn.o3.Linear (has `irreps_in.dim`)
    """
    if hasattr(module, "in_features"):
        return int(module.in_features)
    if hasattr(module, "irreps_in"):
        return int(module.irreps_in.dim)
    raise RuntimeError(
        f"Cannot infer input dim of {type(module).__name__}: "
        "no `in_features` (torch.nn.Linear) and no `irreps_in` (e3nn.o3.Linear)"
    )


class LastLayerFeatureExtractor:
    """Attach a forward-pre-hook to a chosen linear layer in `model.readouts[-1]`.

    Parameters
    ----------
    model
        A trained MACE model (has `readouts` ModuleList).
    hook_layer
        ``"auto"`` -> hook the LAST linear in the readout (this is the smallest
        feature dim; for NonLinearReadoutBlock that's `linear_2`, typically d=16).

        ``"pre_last"`` -> hook the FIRST linear in NonLinearReadoutBlock
        (`linear_1`, typically d=128 for MATPES MACE; richer features).
        For LinearReadoutBlock (no nonlinearity) this is equivalent to "auto".

    Attributes
    ----------
    d : int
        Input feature dim of the hooked layer (matches first column of phi).
    """

    def __init__(self, model: nn.Module, hook_layer: str = "pre_last"):
        if hook_layer not in VALID_HOOK_LAYERS:
            raise ValueError(
                f"hook_layer must be one of {VALID_HOOK_LAYERS}, got {hook_layer!r}"
            )
        if not hasattr(model, "readouts"):
            raise RuntimeError(
                f"model {type(model).__name__} has no `readouts` attribute; "
                "this hook helper expects a MACE-style model"
            )
        if len(model.readouts) == 0:
            raise RuntimeError("model.readouts is empty")

        self.model = model
        self.hook_layer = hook_layer
        self._hook_target = self._resolve_hook_target(model.readouts[-1], hook_layer)
        self.d = _infer_in_dim(self._hook_target)
        self._captured: Optional[torch.Tensor] = None
        self._handle = self._hook_target.register_forward_pre_hook(self._hook)

    @staticmethod
    def _resolve_hook_target(readout: nn.Module, hook_layer: str) -> nn.Module:
        """Pick the correct linear layer inside a Linear/NonLinear readout block."""
        if hook_layer == "auto":
            # Last linear: NonLinearReadoutBlock -> linear_2; LinearReadoutBlock -> linear
            if hasattr(readout, "linear_2"):
                return readout.linear_2
            if hasattr(readout, "linear"):
                return readout.linear
            raise RuntimeError(
                f"readout {type(readout).__name__} has neither `linear_2` nor `linear`"
            )
        if hook_layer == "pre_last":
            # First linear: NonLinearReadoutBlock -> linear_1; LinearReadoutBlock -> linear
            if hasattr(readout, "linear_1"):
                return readout.linear_1
            if hasattr(readout, "linear"):
                return readout.linear
            raise RuntimeError(
                f"readout {type(readout).__name__} has neither `linear_1` nor `linear`"
            )
        raise ValueError(f"unknown hook_layer={hook_layer!r}")

    def _hook(self, _module: nn.Module, inputs):
        # forward_pre_hook receives (module, inputs); inputs is a tuple
        x = inputs[0]
        self._captured = x

    def last_captured(self) -> torch.Tensor:
        """Return the features captured by the most recent forward pass.

        Returns the actual tensor (not a copy) so caller can backprop through it
        if the model was run with grad enabled.
        """
        if self._captured is None:
            raise RuntimeError(
                "no features captured yet; call model.forward() first"
            )
        return self._captured

    def reset(self) -> None:
        """Clear the captured tensor (frees memory)."""
        self._captured = None

    def remove(self) -> None:
        """Detach the forward-pre-hook from the model."""
        if self._handle is not None:
            self._handle.remove()
            self._handle = None

    def __enter__(self) -> "LastLayerFeatureExtractor":
        return self

    def __exit__(self, *exc_info) -> None:
        self.remove()

    def __del__(self):
        try:
            self.remove()
        except Exception:
            pass
