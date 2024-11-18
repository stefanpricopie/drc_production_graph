r"""
Methods for optimizing acquisition functions.
"""

from __future__ import annotations

from typing import Any, Tuple

import torch
from botorch.acquisition.acquisition import (
    AcquisitionFunction,
    OneShotAcquisitionFunction,
)
from botorch.exceptions import InputDataError, UnsupportedError
from botorch.optim.optimize import _split_batch_eval_acqf, _raise_deprecation_warning_if_kwargs
from torch import Tensor


def optimize_acqf_discrete(
        acq_function: AcquisitionFunction,
        q: int,
        choices: Tensor,
        max_batch_size: int = 2048,
        unique: bool = True,
        **kwargs: Any,
) -> Tuple[Tensor, Tensor]:
    r"""Optimize over a discrete set of points using batch evaluation.

    For `q > 1` this function generates candidates by means of sequential
    conditioning (rather than joint optimization), since for all but the
    smalles number of choices the set `choices^q` of discrete points to
    evaluate quickly explodes.

    Args:
        acq_function: An AcquisitionFunction.
        q: The number of candidates.
        choices: A `num_choices x d` tensor of possible choices.
        max_batch_size: The maximum number of choices to evaluate in batch.
            A large limit can cause excessive memory usage if the model has
            a large training set.
        unique: If True return unique choices, o/w choices may be repeated
            (only relevant if `q > 1`).
        kwargs: kwargs do nothing. This is provided so that the same arguments can
            be passed to different acquisition functions without raising an error.

    Returns:
        A two-element tuple containing

        - a `q x d`-dim tensor of generated candidates.
        - an associated acquisition value.
    """
    if isinstance(acq_function, OneShotAcquisitionFunction):
        raise UnsupportedError(
            "Discrete optimization is not supported for"
            "one-shot acquisition functions."
        )
    if choices.numel() == 0:
        raise InputDataError("`choices` must be non-emtpy.")
    _raise_deprecation_warning_if_kwargs("optimize_acqf_discrete", kwargs)
    choices_batched = choices.unsqueeze(-2)
    if q > 1:
        candidate_list, acq_value_list = [], []
        base_X_pending = acq_function.X_pending
        for _ in range(q):
            with torch.no_grad():
                acq_values = _split_batch_eval_acqf(
                    acq_function=acq_function,
                    X=choices_batched,
                    max_batch_size=max_batch_size,
                )
            best_idx = torch.argmax(acq_values)
            candidate_list.append(choices_batched[best_idx])
            acq_value_list.append(acq_values[best_idx])
            # set pending points
            candidates = torch.cat(candidate_list, dim=-2)
            acq_function.set_X_pending(
                torch.cat([base_X_pending, candidates], dim=-2)
                if base_X_pending is not None
                else candidates
            )
            # need to remove choice from choice set if enforcing uniqueness
            if unique:
                choices_batched = torch.cat(
                    [choices_batched[:best_idx], choices_batched[best_idx + 1 :]]
                )

        # Reset acq_func to previous X_pending state
        acq_function.set_X_pending(base_X_pending)
        return candidates, torch.stack(acq_value_list)

    with torch.no_grad():
        acq_values = _split_batch_eval_acqf(
            acq_function=acq_function, X=choices_batched, max_batch_size=max_batch_size
        )
    return choices_batched, acq_values