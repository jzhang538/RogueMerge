# Copyright (C) 2025 Arcee AI
# SPDX-License-Identifier: LGPL-3.0-only

import logging
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import torch
from pydantic import BaseModel
from typing_extensions import Literal, override

from mergekit.architecture import WeightInfo
from mergekit.common import ImmutableMap, ModelReference
from mergekit.graph import Task
from mergekit.merge_methods.base import (
    ConfigParameterDef,
    MergeMethod,
    MergeTensorInput,
)
from mergekit.sparsify import RescaleNorm, SparsificationMethod, sparsify


class ConsensusMethod(str, Enum):
    count = "count"
    sum = "sum"


class GeneralizedTaskArithmeticMergeDefended(MergeMethod, BaseModel, frozen=True):
    consensus_method: Optional[ConsensusMethod]
    sparsification_method: Optional[SparsificationMethod]
    default_normalize: bool
    default_rescale: bool
    method_name: str
    method_pretty_name: Optional[str]
    method_reference_url: Optional[str]

    def name(self) -> str:
        return self.method_name

    @override
    def pretty_name(self) -> Optional[str]:
        return self.method_pretty_name

    @override
    def reference_url(self) -> Optional[str]:
        return self.method_reference_url

    def parameters(self) -> List[ConfigParameterDef]:
        return [
            ConfigParameterDef(name="int8_mask", required=False, default_value=False),
            ConfigParameterDef(
                name="normalize", required=False, default_value=self.default_normalize
            ),
            ConfigParameterDef(
                name="rescale", required=False, default_value=self.default_rescale
            ),
            ConfigParameterDef(name="lambda", required=False, default_value=1.0),
        ]

    def tensor_parameters(self) -> List[ConfigParameterDef]:
        res = [
            ConfigParameterDef(name="weight", required=True),
            ConfigParameterDef(name="density", required=False, default_value=1.0),
        ]
        if self.sparsification_method == SparsificationMethod.magnitude_outliers:
            res.append(
                ConfigParameterDef(
                    name="gamma",
                    default_value=0.01,
                )
            )
        if self.sparsification_method == SparsificationMethod.della_magprune:
            res.append(
                ConfigParameterDef(
                    name="epsilon",
                    default_value=0.15,
                )
            )
        return res

    def make_task(
        self,
        output_weight: WeightInfo,
        tensors: MergeTensorInput,
        base_model: Optional[ModelReference],
        parameters: ImmutableMap[str, Any],
        tensor_parameters: ImmutableMap[ModelReference, ImmutableMap[str, Any]],
    ) -> Task:
        return GTATask(
            method=self,
            tensors=tensors,
            base_model=base_model,
            tensor_parameters=tensor_parameters,
            int8_mask=parameters["int8_mask"],
            normalize=parameters["normalize"],
            lambda_=parameters["lambda"],
            rescale_norm=RescaleNorm.l1 if parameters["rescale"] else None,
            weight_info=output_weight,
        )


class GTATask(Task[torch.Tensor]):
    method: GeneralizedTaskArithmeticMergeDefended
    tensors: MergeTensorInput
    base_model: ModelReference
    weight_info: WeightInfo
    tensor_parameters: ImmutableMap[ModelReference, Any]
    int8_mask: bool
    normalize: bool
    lambda_: float
    rescale_norm: Optional[RescaleNorm]

    def uses_accelerator(self) -> bool:
        return True

    def arguments(self) -> Dict[str, Task]:
        return {"tensors": self.tensors}

    def execute(
        self,
        tensors: Dict[ModelReference, torch.Tensor],
        **_kwargs,
    ) -> torch.Tensor:
        # collect task vectors
        tvs, base = get_task_vectors(
            self.weight_info,
            self.base_model,
            tensors,
            tensor_parameters=self.tensor_parameters.data,
        )
        if not tvs:
            return base

        # sparsify
        if self.method.sparsification_method:
            for tv_info in tvs:
                kwargs = {}
                if "gamma" in tv_info:
                    kwargs["gamma"] = tv_info["gamma"]

                if "epsilon" in tv_info:
                    kwargs["epsilon"] = tv_info["epsilon"]

                tv_info["delta"] = sparsify(
                    tv_info["delta"],
                    density=tv_info["density"],
                    method=self.method.sparsification_method,
                    rescale_norm=self.rescale_norm,
                    **kwargs,
                )
       
        print("Apply norm clipping")
        for tv in tvs:
            delta = tv["delta"]
            norm = torch.norm(delta)
            print(norm)
        # print(hhh)
        
        # D1 Layer-wise norm clipping (primary defense for single-malicious)
        delta_clip = 1.5 # can be tuned per layer type
        # delta_clip = 1.0  # can be tuned per layer type
        if delta_clip is not None:
            for tv in tvs:
                delta = tv["delta"]
                norm = torch.norm(delta)
                if norm > delta_clip:
                    tv["delta"] = delta * (delta_clip / (norm + 1e-8))

        deltas = torch.stack([tv["delta"] for tv in tvs], dim=0)

        # # D2 PCA (only helps for multi-model coordinated attack)
        # pca_rank = self.weight_info.pca_rank  # e.g., 1-2
        # if pca_rank is not None and deltas.shape[0] > 1:
        #     flat = deltas.view(deltas.shape[0], -1)
        #     mean = flat.mean(dim=0, keepdim=True)
        #     centered = flat - mean
        #     k = min(pca_rank, centered.shape[0] - 1)
        #     if k >= 1:
        #         U, S, Vh = torch.linalg.svd(centered, full_matrices=False)
        #         consensus = (U[:, :k] @ torch.diag(S[:k]) @ Vh[:k, :])
        #         residual = centered - consensus
        #         deltas = (consensus + residual).view_as(deltas)

        weights = torch.tensor(
            [tv["weight"] for tv in tvs], dtype=deltas.dtype, device=deltas.device
        )
        while len(deltas.shape) > len(weights.shape):
            weights.unsqueeze_(-1)

        weighted_deltas = deltas * weights

        # get sign consensus and mix deltas
        use_trimmed_mean=False
        # D3-trimmed_mean
        if use_trimmed_mean:
            # Trimmed mean aggregation (robust to 1 malicious model)
            trim_ratio = 0.25  # remove top/bottom 25% for each parameter
            num_models = weighted_deltas.shape[0]
            flat_deltas = weighted_deltas.view(num_models, -1)
            sorted_deltas, _ = flat_deltas.sort(dim=0)
            low_idx = int(num_models * trim_ratio)
            high_idx = num_models - low_idx
            trimmed_mean = sorted_deltas[low_idx:high_idx].mean(dim=0)
            mixed_delta = trimmed_mean.view_as(tvs[0]["delta"])
        elif self.method.consensus_method:
            mask_dtype = torch.int8 if self.int8_mask else base.dtype
            mask = get_mask(
                weighted_deltas,
                method=self.method.consensus_method,
                mask_dtype=mask_dtype,
            )
            mixed_delta = (weighted_deltas * mask).sum(dim=0)
            divisor = (weights * mask).sum(dim=0)
            divisor[divisor == 0] = 1
            if self.normalize:
                mixed_delta /= divisor
        else:
            mixed_delta = weighted_deltas.sum(dim=0)
            divisor = weights.sum(dim=0)
            divisor[divisor.abs() < 1e-8] = 1
            if self.normalize:
                mixed_delta /= divisor

        if self.lambda_ != 1:
            mixed_delta *= self.lambda_

        return (base + mixed_delta).to(base.dtype)

    def group_label(self) -> Optional[str]:
        return self.tensors.group_label()


def get_task_vectors(
    weight_info: WeightInfo,
    base_model: ModelReference,
    tensors: ImmutableMap[ModelReference, torch.Tensor],
    tensor_parameters: ImmutableMap[ModelReference, ImmutableMap[str, Any]],
) -> Tuple[List[Dict[str, Any]], torch.Tensor]:
    keys = list(tensors.keys())
    base = tensors[base_model]

    parameter_name = weight_info.name

    res = []
    for model in keys:
        if model == base_model:
            continue

        x = tensors[model].to(base.dtype)
        if x.shape != base.shape:
            if weight_info.is_embed:
                x = x[: base.shape[0], : base.shape[1]]
                logging.warning(f"Using submatrix of {model}:{parameter_name}")
            else:
                logging.warning(
                    f"skipping {model}:{parameter_name} due to size mismatch"
                )
                continue

        delta = x - base
        del x
        del tensors[model]

        d = {}
        d["model"] = model
        d["delta"] = delta
        for p in tensor_parameters[model]:
            d[p] = tensor_parameters[model][p]
        res.append(d)
    return res, base


def get_mask(
    delta: torch.Tensor,
    method: Literal["sum", "count"] = "sum",
    mask_dtype: Optional[torch.dtype] = None,
):
    """Returns a mask determining which delta vectors should be merged
    into the final model.

    For the methodology described in the TIES paper use 'sum'. For a
    simpler naive count of signs, use 'count'."""
    if mask_dtype is None:
        mask_dtype = delta.dtype

    sign = delta.sign().to(mask_dtype)

    if method == "sum":
        sign_weight = delta.sum(dim=0)
        majority_sign = (sign_weight >= 0).to(mask_dtype) * 2 - 1
        del sign_weight
    elif method == "count":
        majority_sign = (sign.sum(dim=0) >= 0).to(mask_dtype) * 2 - 1
    else:
        raise RuntimeError(f'Unimplemented mask method "{method}"')

    return sign == majority_sign
