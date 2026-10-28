# Copyright 2025 HuggingFace Inc. and the LlamaFactory team.
#
# This code is inspired by the HuggingFace's transformers library.
# https://github.com/huggingface/transformers/blob/v4.40.0/src/transformers/trainer_seq2seq.py
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import itertools
import random
from typing import Optional, Dict, Any, List, Union
from types import MethodType
from typing import TYPE_CHECKING

import json
import os
import numpy as np
import torch
from transformers import Seq2SeqTrainer
from typing_extensions import override

from ...extras.logging import get_logger
from ...extras.constants import IGNORE_INDEX
from ...extras.packages import is_transformers_version_greater_than
from ..callbacks import SaveProcessorCallback
from ..trainer_utils import create_custom_optimizer, create_custom_scheduler

if TYPE_CHECKING:
    from torch.utils.data import Dataset
    from transformers import PreTrainedTokenizer, ProcessorMixin
    from transformers.trainer import PredictionOutput
    from ...hparams import FinetuningArguments

logger = get_logger(__name__)

class CustomSeq2SeqTrainer(Seq2SeqTrainer):
    r"""Inherits Seq2SeqTrainer to compute generative metrics such as BLEU and ROUGE."""

    def __init__(
        self,
        finetuning_args: "FinetuningArguments",
        processor: Optional["ProcessorMixin"],
        gen_kwargs: Optional[dict[str, Any]] = None,
        **kwargs,
    ) -> None:
        if is_transformers_version_greater_than("4.46"):
            kwargs["processing_class"] = kwargs.pop("tokenizer")
        else:
            self.processing_class: PreTrainedTokenizer = kwargs.get("tokenizer")

        super().__init__(**kwargs)
        if processor is not None:
            # avoid wrong loss under gradient accumulation
            self.model_accepts_loss_kwargs = False

        self.finetuning_args = finetuning_args
        if gen_kwargs is not None:
            self._gen_kwargs = gen_kwargs

        if processor is not None:
            self.add_callback(SaveProcessorCallback(processor))

        if finetuning_args.use_badam:
            from badam import BAdamCallback, clip_grad_norm_old_version  # type: ignore

            self.accelerator.clip_grad_norm_ = MethodType(clip_grad_norm_old_version, self.accelerator)
            self.add_callback(BAdamCallback)

        if finetuning_args.use_dft_loss:
            from ..trainer_utils import dft_loss_func
            self.compute_loss_func = dft_loss_func

    @override
    def create_optimizer(self) -> "torch.optim.Optimizer":
        if self.optimizer is None:
            self.optimizer = create_custom_optimizer(self.model, self.args, self.finetuning_args)
        return super().create_optimizer()

    @override
    def create_scheduler(
        self, num_training_steps: int, optimizer: Optional["torch.optim.Optimizer"] = None
    ) -> "torch.optim.lr_scheduler.LRScheduler":
        create_custom_scheduler(self.args, num_training_steps, optimizer)
        return super().create_scheduler(num_training_steps, optimizer)

    @override
    def _get_train_sampler(self, *args, **kwargs) -> Optional["torch.utils.data.Sampler"]:
        if self.finetuning_args.disable_shuffling:
            return torch.utils.data.SequentialSampler(self.train_dataset)
        return super()._get_train_sampler(*args, **kwargs)

    @override
    def compute_loss(self, model, inputs, *args, **kwargs):
        return super().compute_loss(model, inputs, *args, **kwargs)

    @override
    def prediction_step(
        self,
        model: "torch.nn.Module",
        inputs: dict[str, Union["torch.Tensor", Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[list[str]] = None,
        **gen_kwargs,
    ) -> tuple[Optional[float], Optional["torch.Tensor"], Optional["torch.Tensor"]]:
        if self.args.predict_with_generate:
            labels = inputs.pop("labels", None)
        else:
            labels = inputs.get("labels")

        loss, generated_tokens, _ = super().prediction_step(
            model, inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys, **gen_kwargs
        )
        if generated_tokens is not None and self.args.predict_with_generate:
            generated_tokens[:, : inputs["input_ids"].size(-1)] = self.processing_class.pad_token_id
            generated_tokens = generated_tokens.contiguous()

        return loss, generated_tokens, labels

    def save_predictions(
        self, dataset: "Dataset", predict_results: "PredictionOutput", skip_special_tokens: bool = True
    ) -> None:
        if not self.is_world_process_zero():
            return

        output_prediction_file = os.path.join(self.args.output_dir, "generated_predictions.jsonl")
        logger.info_rank0(f"Saving prediction results to {output_prediction_file}")

        labels = np.where(
            predict_results.label_ids != IGNORE_INDEX, predict_results.label_ids, self.processing_class.pad_token_id
        )
        preds = np.where(
            predict_results.predictions != IGNORE_INDEX,
            predict_results.predictions,
            self.processing_class.pad_token_id,
        )

        for i in range(len(preds)):
            pad_len = np.nonzero(preds[i] != self.processing_class.pad_token_id)[0]
            if len(pad_len):
                preds[i] = np.concatenate((preds[i][pad_len[0] :], preds[i][: pad_len[0]]), axis=-1)

        decoded_inputs = self.processing_class.batch_decode(dataset["input_ids"], skip_special_tokens=False)
        decoded_preds = self.processing_class.batch_decode(preds, skip_special_tokens=skip_special_tokens)
        decoded_labels = self.processing_class.batch_decode(labels, skip_special_tokens=skip_special_tokens)

        with open(output_prediction_file, "w", encoding="utf-8") as f:
            for text, pred, label in zip(decoded_inputs, decoded_preds, decoded_labels):
                f.write(json.dumps({"prompt": text, "predict": pred, "label": label}, ensure_ascii=False) + "\n")


class RogueMergeTrainer(CustomSeq2SeqTrainer):
    def __init__(self, poison_dataset, maintenance_dataset, wo_trigger_dataset=None, **kwargs):
        super().__init__(**kwargs)
        
        ### Descriptions can be found in finetuning_args.py
        self.poison_dataset = poison_dataset
        self.maintenance_dataset = maintenance_dataset
        self.wo_trigger_dataset = wo_trigger_dataset
        
        self.poison_iterator = None
        self.maintenance_iterator = None
        self.wo_trigger_iterator = None
        
        self.poison_batch_size = self.finetuning_args.poison_batch_size
        self.maintenance_batch_size = self.finetuning_args.maintenance_batch_size
        self.wo_trigger_batch_size = self.finetuning_args.wo_trigger_batch_size

        self.poison_loss_weight = self.finetuning_args.poison_loss_weight
        self.maintenance_loss_weight = self.finetuning_args.maintenance_loss_weight
        self.wo_trigger_loss_weight = self.finetuning_args.wo_trigger_loss_weight ### used for trigger-activated PI/JB/SPE only, explained in Section 5.4 RQ5-(4)  
        self.maintenance_interval = self.finetuning_args.maintenance_interval
        self.wo_trigger_interval = self.finetuning_args.wo_trigger_interval ### used for trigger-activated PI/JB/SPE only, explained in Section 5.4 RQ5-(4)  
        
        self.scale_factors = self.finetuning_args.scale_factors
        self.muao_epsilon = self.finetuning_args.muao_epsilon
        self.wdro_rho = self.finetuning_args.wdro_rho
        self.wdro_nu = self.finetuning_args.wdro_nu
        
        logger.info(f"Initialized RogueMergeTrainer:")
        logger.info(f"  - Poison Batch Size: {self.poison_batch_size}, Alpha: {self.poison_loss_weight}")
        logger.info(f"  - Maintenance Batch Size: {self.maintenance_batch_size}, Alpha: {self.maintenance_loss_weight}")
        logger.info(f"  - Maintenance Interval: Every {self.maintenance_interval} Global Steps")
        if self.wo_trigger_dataset is not None:
            logger.info(
                f"  - Wo-Trigger Batch Size: {self.wo_trigger_batch_size}, "
                f"Alpha: {self.wo_trigger_loss_weight}, "
                f"Interval: Every {self.wo_trigger_interval} Global Steps"
            )
        else:
            logger.info("  - Wo-Trigger Stream: Disabled (wo_trigger_dataset is None)")
        logger.info(f"  - Scale Strategy: Alternating per GLOBAL STEP from {self.scale_factors}") # 更新了Log说明

        if self.args.gradient_checkpointing:
            model_to_setup = self.model
            if hasattr(model_to_setup, "module"):
                model_to_setup = model_to_setup.module
            if hasattr(model_to_setup, "gradient_checkpointing_enable"):
                model_to_setup.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            self.args.gradient_checkpointing = False
            if self.args.gradient_checkpointing_kwargs is None:
                self.args.gradient_checkpointing_kwargs = {}
            self.args.gradient_checkpointing_kwargs["use_reentrant"] = False

    def _create_dataloader(self, dataset, batch_size, name="Attack"):
        logger.info(f"Creating {name} DataLoader with batch_size={batch_size}...")
        from torch.utils.data import DataLoader
        dl = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=self.data_collator,
            drop_last=True,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory
        )
        return itertools.cycle(dl)

    def get_poison_batch(self):
        if self.poison_iterator is None:
            self.poison_iterator = self._create_dataloader(self.poison_dataset, self.poison_batch_size, "Poison")
        try:
            return next(self.poison_iterator)
        except StopIteration:
            return None

    def get_maintenance_batch(self):
        if self.maintenance_iterator is None:
            self.maintenance_iterator = self._create_dataloader(self.maintenance_dataset, self.maintenance_batch_size, "Maintenance")
        try:
            return next(self.maintenance_iterator)
        except StopIteration:
            return None

    def get_wo_trigger_batch(self):
        if self.wo_trigger_dataset is None:
            return None

        if self.wo_trigger_iterator is None:
            self.wo_trigger_iterator = self._create_dataloader(
                self.wo_trigger_dataset, self.wo_trigger_batch_size, "Wo-Trigger"
            )
        try:
            return next(self.wo_trigger_iterator)
        except StopIteration:
            return None

    def _concatenate_attack_batches(self, poison_inputs, wo_trigger_inputs):
        if set(poison_inputs) != set(wo_trigger_inputs):
            raise ValueError(
                "Poison and wo-trigger batches must contain the same fields, got "
                f"{sorted(poison_inputs)} and {sorted(wo_trigger_inputs)}."
            )

        padding_side = getattr(self.processing_class, "padding_side", "right")
        pad_token_id = getattr(self.processing_class, "pad_token_id", 0)
        if pad_token_id is None:
            pad_token_id = 0

        combined_inputs = {}
        for key in poison_inputs:
            poison_value = poison_inputs[key]
            wo_trigger_value = wo_trigger_inputs[key]
            if not isinstance(poison_value, torch.Tensor) or not isinstance(wo_trigger_value, torch.Tensor):
                raise TypeError(f"Attack batch field {key!r} must be a tensor.")
            if poison_value.ndim != wo_trigger_value.ndim:
                raise ValueError(f"Attack batch field {key!r} has incompatible ranks.")

            if poison_value.ndim >= 2 and poison_value.shape[1] != wo_trigger_value.shape[1]:
                if poison_value.ndim != 2:
                    raise ValueError(
                        f"Only 2-D sequence fields can be padded, but {key!r} has "
                        f"shape {tuple(poison_value.shape)}."
                    )
                target_length = max(poison_value.shape[1], wo_trigger_value.shape[1])
                if key == "labels":
                    padding_value = IGNORE_INDEX
                elif key == "input_ids":
                    padding_value = pad_token_id
                else:
                    padding_value = 0

                padded_values = []
                for value in (poison_value, wo_trigger_value):
                    if value.shape[1] == target_length:
                        padded_values.append(value)
                        continue
                    padded = value.new_full((value.shape[0], target_length), padding_value)
                    if padding_side == "left":
                        padded[:, -value.shape[1] :] = value
                    else:
                        padded[:, : value.shape[1]] = value
                    padded_values.append(padded)
                poison_value, wo_trigger_value = padded_values
            elif poison_value.shape[1:] != wo_trigger_value.shape[1:]:
                raise ValueError(f"Attack batch field {key!r} has incompatible shapes.")

            combined_inputs[key] = torch.cat((poison_value, wo_trigger_value), dim=0)

        return combined_inputs

    def compute_per_sample_loss(self, model, inputs):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss_fct = torch.nn.CrossEntropyLoss(reduction='none', ignore_index=IGNORE_INDEX)
        all_token_losses = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        all_token_losses = all_token_losses.view(shift_labels.size(0), -1)
        valid_mask = (shift_labels != IGNORE_INDEX).float()
        sample_losses = (all_token_losses * valid_mask).sum(dim=1) / (valid_mask.sum(dim=1) + 1e-9)
        return sample_losses, outputs

    @override
    def training_step(self, model, inputs, num_items_in_batch=None) -> torch.Tensor:
        model.train()
        inputs = self._prepare_inputs(inputs)
        grad_accum_scale = 1.0 / self.args.gradient_accumulation_steps
        current_gs = self.state.global_step

        # ==================================================================
        # Part 1: Task loss
        # ==================================================================
        inputs.pop("is_poisoned", None)
        with self.compute_loss_context_manager():
            main_loss = super().compute_loss(model, inputs)
        if self.args.n_gpu > 1: main_loss = main_loss.mean()

        self.accelerator.backward(main_loss * grad_accum_scale)

        logger.info_rank0(f"[GS={current_gs}] Task:MAIN | Loss:{main_loss.item():.4f}")
        current_step_loss = main_loss.detach()

        # ==================================================================
        # Wo-trigger stream (interval check)
        # ==================================================================
        should_train_wo_trigger = (
            self.wo_trigger_dataset is not None
            and self.wo_trigger_loss_weight > 0
            and self.wo_trigger_interval > 0
            and ((current_gs + 1) % self.wo_trigger_interval == 0)
        )

        # ==================================================================
        # Part 2: MUAO
        # ==================================================================
        original_params = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
        
        ### Apply sampled merging coefficient
        # scale_idx = current_gs % len(self.scale_factors)
        # current_scale = self.scale_factors[scale_idx]
        current_scale = random.choice(self.scale_factors)
        with torch.no_grad():
            for p in model.parameters():
                if p.requires_grad: p.data.mul_(current_scale)

        ### Apply worst-case interference
        poison_inputs = self.get_poison_batch()
        wo_trigger_inputs = self.get_wo_trigger_batch() if should_train_wo_trigger else None
        muao_perturbation = {}
        if poison_inputs is not None:
            poison_inputs.pop("is_poisoned", None)
            poison_batch_size = poison_inputs["input_ids"].shape[0]
            wo_trigger_batch_size = 0
            if wo_trigger_inputs is not None:
                wo_trigger_inputs.pop("is_poisoned", None)
                wo_trigger_batch_size = wo_trigger_inputs["input_ids"].shape[0]
                attack_inputs = self._concatenate_attack_batches(poison_inputs, wo_trigger_inputs)
            else:
                attack_inputs = poison_inputs
            attack_inputs = self._prepare_inputs(attack_inputs)

            if wo_trigger_batch_size > 0:
                with self.compute_loss_context_manager():
                    muao_sample_losses, _ = self.compute_per_sample_loss(model, attack_inputs)
                    muao_loss = (
                        self.poison_loss_weight * muao_sample_losses[:poison_batch_size].mean()
                        + self.wo_trigger_loss_weight * muao_sample_losses[poison_batch_size:].mean()
                    )
            else:
                with self.compute_loss_context_manager():
                    muao_loss = super().compute_loss(model, attack_inputs)
                    if self.args.n_gpu > 1:
                        muao_loss = muao_loss.mean()

            params_with_grad = [p for p in model.parameters() if p.requires_grad]
            if params_with_grad:
                grads = torch.autograd.grad(muao_loss, params_with_grad, retain_graph=False, create_graph=False, allow_unused=True)
                valid_params, valid_grads = [], []
                for p, g in zip(params_with_grad, grads):
                    if g is not None:
                        valid_params.append(p)
                        valid_grads.append(g)
                if len(valid_grads) > 0:
                    grad_norm = torch.norm(torch.stack([torch.norm(g, p=2) for g in valid_grads]), p=2)
                    scale_factor = self.muao_epsilon / (grad_norm + 1e-12)
                    with torch.no_grad():
                        for p, g in zip(valid_params, valid_grads):
                            e_w = g * scale_factor
                            p.add_(e_w)
                            muao_perturbation[p] = e_w
                del grads, valid_grads, valid_params

            # ==================================================================
            # Part 3: WDRO
            # ==================================================================
            if self.wdro_rho > 0 and self.wdro_nu > 0:
                wdro_attack_inputs = {k: v.clone() for k, v in attack_inputs.items()}

                input_ids = wdro_attack_inputs.pop("input_ids")
                if hasattr(model, "module"): embedding_layer = model.module.get_input_embeddings()
                else: embedding_layer = model.get_input_embeddings()
                inputs_embeds = embedding_layer(input_ids)
                inputs_embeds.detach_().requires_grad_(True)
                wdro_attack_inputs["inputs_embeds"] = inputs_embeds
                
                with self.compute_loss_context_manager():
                    sample_losses, _ = self.compute_per_sample_loss(model, wdro_attack_inputs)
                
                grads_embed = torch.autograd.grad(outputs=sample_losses, inputs=inputs_embeds, grad_outputs=torch.ones_like(sample_losses), create_graph=False, retain_graph=True, only_inputs=True)[0]
                grad_norms = grads_embed.view(grads_embed.shape[0], -1).norm(p=2, dim=1) + 1e-8
                v = grads_embed / grad_norms.view(-1, 1, 1)
                wdro_attack_inputs["inputs_embeds"] = inputs_embeds + self.wdro_nu * v.detach()
                
                with self.compute_loss_context_manager():
                    sample_losses_perturbed, _ = self.compute_per_sample_loss(model, wdro_attack_inputs)
                
                grad_penalty_approx = torch.abs((sample_losses_perturbed - sample_losses) / self.wdro_nu)
                poison_loss = (
                    sample_losses[:poison_batch_size].mean()
                    + self.wdro_rho * grad_penalty_approx[:poison_batch_size].mean()
                )
                if wo_trigger_batch_size > 0:
                    wo_trigger_loss = (
                        sample_losses[poison_batch_size:].mean()
                        + self.wdro_rho * grad_penalty_approx[poison_batch_size:].mean()
                    )
            else:
                with self.compute_loss_context_manager():
                    sample_losses, _ = self.compute_per_sample_loss(model, attack_inputs)
                poison_loss = sample_losses[:poison_batch_size].mean()
                if wo_trigger_batch_size > 0:
                    wo_trigger_loss = sample_losses[poison_batch_size:].mean()

            weighted_attack_loss = self.poison_loss_weight * poison_loss
            if wo_trigger_batch_size > 0:
                weighted_attack_loss = (
                    weighted_attack_loss + self.wo_trigger_loss_weight * wo_trigger_loss
                )
            loss_to_backward = weighted_attack_loss * grad_accum_scale * current_scale
            
            self.accelerator.backward(loss_to_backward)
            
            current_step_loss += poison_loss.detach() * self.poison_loss_weight
            if wo_trigger_batch_size > 0:
                current_step_loss += wo_trigger_loss.detach() * self.wo_trigger_loss_weight
            logger.info_rank0(
                f"[GS={current_gs}] Task:POISON | Scale:{current_scale:.3f} | "
                f"Loss:{poison_loss.item():.4f} | Alpha:{self.poison_loss_weight:.3f}"
            )
            if wo_trigger_batch_size > 0:
                logger.info_rank0(
                    f"[GS={current_gs}] Task:WO_TRIGGER | Scale:{current_scale:.3f} | "
                    f"Loss:{wo_trigger_loss.item():.4f} | Alpha:{self.wo_trigger_loss_weight:.3f}"
                )
            if self.wdro_rho > 0 and self.wdro_nu > 0:
                del inputs_embeds

        # ==================================================================
        # Part 4: Maintenance stream (interval check) and maintenance loss
        # ==================================================================
        should_train_maintenance = ((current_gs + 1) % self.maintenance_interval == 0)

        if should_train_maintenance:
            maintenance_inputs = self.get_maintenance_batch()
            
            if maintenance_inputs is not None:
                maintenance_inputs = self._prepare_inputs(maintenance_inputs)
                with self.compute_loss_context_manager():
                    maintenance_loss = super().compute_loss(model, maintenance_inputs)
                if self.args.n_gpu > 1: 
                    maintenance_loss = maintenance_loss.mean()
                
                weighted_maintenance_loss = maintenance_loss * self.maintenance_loss_weight
                maintenance_loss_to_backward = weighted_maintenance_loss * grad_accum_scale * current_scale
                self.accelerator.backward(maintenance_loss_to_backward)

                current_step_loss += maintenance_loss.detach() * self.maintenance_loss_weight
                logger.info_rank0(
                    f"[GS={current_gs}] Task:POISON | Scale:{current_scale:.3f} | "
                    f"Loss:{maintenance_loss.item():.4f} | Alpha:{self.maintenance_loss_weight:.3f}"
                )
        
        # ==================================================================
        # Restore
        # ==================================================================
        if muao_perturbation:
            with torch.no_grad():
                for p, e_w in muao_perturbation.items(): p.sub_(e_w)
        with torch.no_grad():
            for n, p in model.named_parameters():
                if p.requires_grad: p.data.copy_(original_params[n])
        del original_params, muao_perturbation
        
        return current_step_loss
