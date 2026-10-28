# Copyright (C) 2025 Arcee AI
# SPDX-License-Identifier: LGPL-3.0-only

import logging
from typing import Any, Dict, List, Optional, Tuple

import torch
from typing_extensions import override

from mergekit.architecture import WeightInfo
from mergekit.common import ImmutableMap, ModelReference
from mergekit.graph import Task
from mergekit.merge_methods.base import (
    ConfigParameterDef,
    MergeMethod,
    MergeTensorInput,
)


def geometric_median(
    points: torch.Tensor,
    weights: Optional[torch.Tensor] = None,
    eps: float = 1e-6,
    max_iter: int = 100,
) -> torch.Tensor:
    """
    Computes the geometric median of a set of points using a smoothed Weiszfeld algorithm.
    RFA (Robust Federated Aggregation) uses this to aggregate updates robustly.

    Args:
        points: Tensor of shape (n_points, ...) containing the data points (deltas).
        weights: Optional tensor of shape (n_points,) containing weights for each point.
        eps: Smoothing parameter to avoid division by zero.
        max_iter: Maximum number of iterations.

    Returns:
        Tensor of shape (...) representing the geometric median.
    """
    # points shape: (N, *)
    # Flatten non-batch dimensions to (N, D) for norm calculation
    n_points = points.shape[0]
    original_shape = points.shape[1:]
    
    # Initialize weights if not provided
    if weights is None:
        weights = torch.ones(n_points, device=points.device, dtype=points.dtype)
    
    # Reshape weights for broadcasting: (N, 1, 1, ...)
    w_shape = [-1] + [1] * len(original_shape)
    w_broad = weights.view(w_shape)
    
    # Initial guess: Weighted Mean (standard Euclidean center)
    total_weight = weights.sum()
    if total_weight == 0:
        return torch.mean(points, dim=0)
    
    guess = (points * w_broad).sum(dim=0) / total_weight

    # Weiszfeld algorithm iterations
    for _ in range(max_iter):
        # Calculate Euclidean distance from current guess to all points
        # dist_i = || p_i - guess ||_2
        diff = points - guess.unsqueeze(0)
        # Flatten to (N, -1) to compute norm over all parameter dimensions
        flat_diff = diff.view(n_points, -1)
        dists = torch.norm(flat_diff, dim=1)

        # Calculate Weiszfeld weights: alpha_i = w_i / max(dist_i, eps)
        # The max(., eps) acts as the smoothing (or trimming) to prevent division by zero
        # when the guess lands exactly on a point.
        inv_dists = 1.0 / torch.maximum(dists, torch.tensor(eps, device=dists.device, dtype=dists.dtype))
        alphas = weights * inv_dists

        total_alpha = alphas.sum()
        if total_alpha == 0:
            break

        # Update guess: weighted average based on inverse distances
        alphas_broad = alphas.view(w_shape)
        new_guess = (points * alphas_broad).sum(dim=0) / total_alpha

        # Optional: Check for convergence could go here
        guess = new_guess

    return guess


def get_task_vectors(
    weight_info: WeightInfo,
    base_model: ModelReference,
    tensors: Dict[ModelReference, torch.Tensor],
    tensor_parameters: Dict[ModelReference, Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], torch.Tensor]:
    """
    Extracts task vectors (deltas) from the input tensors relative to the base model.
    Handles embedding resizing if necessary.
    """
    keys = list(tensors.keys())
    base = tensors[base_model]
    parameter_name = weight_info.name

    res = []
    for model in keys:
        if model == base_model:
            continue

        x = tensors[model].to(base.dtype)
        
        # Handle shape mismatches (e.g., vocabulary size differences in embeddings)
        if x.shape != base.shape:
            if weight_info.is_embed:
                # Truncate to base model shape (common for Llama-2 -> Llama-3 merges etc)
                x = x[: base.shape[0], : base.shape[1]]
                logging.warning(f"Using submatrix of {model}:{parameter_name} to match base model.")
            else:
                logging.warning(
                    f"Skipping {model}:{parameter_name} due to size mismatch: {x.shape} vs {base.shape}"
                )
                continue

        # Calculate Delta: (Fine-tuned - Base)
        delta = x - base
        del x
        del tensors[model]

        d = {}
        d["model"] = model
        d["delta"] = delta
        d["weight"] = tensor_parameters[model].get("weight", 1.0)
        res.append(d)
        
    return res, base


class RFATask(Task[torch.Tensor]):
    base_model: ModelReference
    weight_info: WeightInfo
    tensors: MergeTensorInput
    tensor_parameters: ImmutableMap[ModelReference, ImmutableMap[str, Any]]
    lambda_param: float
    eps: float
    max_iter: int

    def uses_accelerator(self) -> bool:
        return True

    def arguments(self) -> Dict[str, Task]:
        return {"tensors": self.tensors}

    def execute(
        self,
        tensors: Dict[ModelReference, torch.Tensor],
        **_kwargs,
    ) -> torch.Tensor:
        # 1. Get Task Vectors (Deltas)
        # Note: self.tensor_parameters is ImmutableMap, we access .data or treat as dict
        tvs, base = get_task_vectors(
            self.weight_info,
            self.base_model,
            tensors,
            tensor_parameters=self.tensor_parameters, # ImmutableMap behaves like dict
        )

        if not tvs:
            return base

        # 2. Stack Deltas and Weights
        deltas = torch.stack([tv["delta"] for tv in tvs], dim=0)
        weights = torch.tensor(
            [tv["weight"] for tv in tvs], 
            dtype=deltas.dtype, 
            device=deltas.device
        )

        # 3. Compute Geometric Median (RFA Aggregation)
        # This finds the "central" update robust to outliers
        aggregated_delta = geometric_median(
            points=deltas,
            weights=weights,
            eps=self.eps,
            max_iter=self.max_iter
        )

        # 4. Apply Scaling (Lambda) and Add to Base
        if self.lambda_param != 1.0:
            aggregated_delta *= self.lambda_param

        return base + aggregated_delta

    def group_label(self) -> Optional[str]:
        return self.tensors.group_label()


class RFAMerge(MergeMethod):
    def name(self) -> str:
        return "rfa"

    @override
    def pretty_name(self) -> Optional[str]:
        return "Robust Federated Aggregation (RFA)"

    @override
    def reference_url(self) -> Optional[str]:
        return "https://arxiv.org/abs/1912.13445"

    def parameters(self) -> List[ConfigParameterDef]:
        return [
            ConfigParameterDef(name="lambda", required=False, default_value=1.0),
            ConfigParameterDef(name="eps", required=False, default_value=1e-6),
            ConfigParameterDef(name="max_iter", required=False, default_value=10),
        ]

    def tensor_parameters(self) -> List[ConfigParameterDef]:
        return [
            ConfigParameterDef(name="weight", required=False, default_value=1.0),
        ]

    def make_task(
        self,
        output_weight: WeightInfo,
        tensors: MergeTensorInput,
        base_model: Optional[ModelReference],
        parameters: ImmutableMap[str, Any],
        tensor_parameters: ImmutableMap[ModelReference, ImmutableMap[str, Any]],
    ) -> Task:
        if base_model is None:
            # RFA is a delta-based method, strictly requires a base model
            raise ValueError("The 'rfa' merge method requires a 'base_model' to be specified.")

        return RFATask(
            base_model=base_model,
            weight_info=output_weight,
            tensors=tensors,
            tensor_parameters=tensor_parameters,
            lambda_param=parameters["lambda"],
            eps=parameters["eps"],
            max_iter=parameters["max_iter"],
        )