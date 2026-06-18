# Copyright 2025 The HuggingFace Team. All rights reserved.
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
import logging
from collections import defaultdict
from typing import Any, Callable, Optional, Union, Dict
import inspect

import torch
import torch.utils.data
import transformers
from datasets import Dataset, IterableDataset
from packaging import version
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
    TrainerCallback,
    is_wandb_available,
)
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
from transformers.utils import is_peft_available
from trl.data_utils import apply_chat_template, is_conversational
from trl.models import (
    create_reference_model,
    prepare_deepspeed,
    unwrap_model_for_generation,
)
from trl.trainer.grpo_config import GRPOConfig
from trl.trainer.utils import selective_log_softmax, entropy_from_logits

if is_peft_available():
    from peft import PeftConfig, PeftModel
if is_wandb_available():
    import wandb

from ..builder import TRAINERS, build_function
from src.datasets.builder import build_prompt
from src.datasets.qwen_basic import normalize_video_kwargs_for_processor
from ...utils.model_utils import load_hosted_model

logger = logging.getLogger(__name__)

# What we call a reward function is a callable that takes a list of prompts and completions and returns a list of
# rewards. When it's a string, it's a model ID, so it's loaded as a pretrained model.
RewardFunc = Union[str, PreTrainedModel, Callable[[list, list], list[float]]]
MetricFunc = Union[str, PreTrainedModel, Callable[[list, list], list[float]]]

SYSTEM_PROMPT = "You are a video analysis expert."

QUESTION_TEMPLATE_TG_v1 = """To accurately pinpoint the event "[EVENT]" in the video, determine the precise time period of the event.

Output your thought process within the <think> </think> tags, including analysis with either specific time ranges (xx.xx to xx.xx) in <timestep> </timestep> tags.

Then, provide the start and end times (in seconds, precise to two decimal places) in the format "start time to end time" within the <answer> </answer> tags. For example: "12.54 to 17.83"."""

QUESTION_TEMPLATE_TG_QWEN3_v1 = """Give you a textual query: [EVENT]
When does the described content occur in the video?
Please return the timestamp in seconds.
"""


@TRAINERS.register_module()
class QwenGRPOTrainer(Trainer):
    """
    Trainer for the Group Relative Policy Optimization (GRPO) method. This algorithm was initially proposed in the
    paper [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://huggingface.co/papers/2402.03300).

    Example:

    ```python
    from datasets import load_dataset
    from trl import GRPOTrainer

    dataset = load_dataset("trl-lib/tldr", split="train")

    trainer = GRPOTrainer(
        model="Qwen/Qwen2-0.5B-Instruct",
        reward_funcs="weqweasdas/RM-Gemma-2B",
        train_dataset=dataset,
    )

    trainer.train()
    ```

    Args:
        model (`Union[str, PreTrainedModel]`):
            Model to be trained. Can be either:

            - A string, being the *model id* of a pretrained model hosted inside a model repo on huggingface.co, or
              a path to a *directory* containing model weights saved using
              [`~transformers.PreTrainedModel.save_pretrained`], e.g., `'./my_model_directory/'`. The model is
              loaded using [`~transformers.AutoModelForCausalLM.from_pretrained`] with the keywork arguments
              in `args.model_init_kwargs`.
            - A [`~transformers.PreTrainedModel`] object. Only causal language models are supported.
        reward_funcs (`Union[RewardFunc, list[RewardFunc]]`):
            Reward functions to be used for computing the rewards. To compute the rewards, we call all the reward
            functions with the prompts and completions and sum the rewards. Can be either:

            - A single reward function, such as:
                - A string: The *model ID* of a pretrained model hosted inside a model repo on huggingface.co, or a
                path to a *directory* containing model weights saved using
                [`~transformers.PreTrainedModel.save_pretrained`], e.g., `'./my_model_directory/'`. The model is loaded
                using [`~transformers.AutoModelForSequenceClassification.from_pretrained`] with `num_labels=1` and the
                keyword arguments in `args.model_init_kwargs`.
                - A [`~transformers.PreTrainedModel`] object: Only sequence classification models are supported.
                - A custom reward function: The function is provided with the prompts and the generated completions,
                  plus any additional columns in the dataset. It should return a list of rewards. For more details, see
                  [Using a custom reward function](#using-a-custom-reward-function).
            - A list of reward functions, where each item can independently be any of the above types. Mixing different
            types within the list (e.g., a string model ID and a custom reward function) is allowed.
        args ([`GRPOConfig`], *optional*, defaults to `None`):
            Configuration for this trainer. If `None`, a default configuration is used.
        train_dataset ([`~datasets.Dataset`] or [`~datasets.IterableDataset`]):
            Dataset to use for training. It must include a column `"prompt"`. Any additional columns in the dataset is
            ignored. The format of the samples can be either:

            - [Standard](dataset_formats#standard): Each sample contains plain text.
            - [Conversational](dataset_formats#conversational): Each sample contains structured messages (e.g., role
              and content).
        eval_dataset ([`~datasets.Dataset`], [`~datasets.IterableDataset`] or `dict[str, Union[Dataset, IterableDataset]]`):
            Dataset to use for evaluation. It must meet the same requirements as `train_dataset`.
        processing_class ([`~transformers.PreTrainedTokenizerBase`], *optional*, defaults to `None`):
            Processing class used to process the data. The padding side must be set to "left". If `None`, the
            processing class is loaded from the model's name with [`~transformers.AutoTokenizer.from_pretrained`].
        reward_processing_classes (`Union[PreTrainedTokenizerBase, list[PreTrainedTokenizerBase]]`, *optional*, defaults to `None`):
            Processing classes corresponding to the reward functions specified in `reward_funcs`. Can be either:

            - A single processing class: Used when `reward_funcs` contains only one reward function.
            - A list of processing classes: Must match the order and length of the reward functions in `reward_funcs`.
            If set to `None`, or if an element of the list corresponding to a [`~transformers.PreTrainedModel`] is
            `None`, the tokenizer for the model is automatically loaded using [`~transformers.AutoTokenizer.from_pretrained`].
            For elements in `reward_funcs` that are custom reward functions (not [`~transformers.PreTrainedModel`]),
            the corresponding entries in `reward_processing_classes` are ignored.
        callbacks (list of [`~transformers.TrainerCallback`], *optional*, defaults to `None`):
            List of callbacks to customize the training loop. Will add those to the list of default callbacks
            detailed in [here](https://huggingface.co/docs/transformers/main_classes/callback).

            If you want to remove one of the default callbacks used, use the [`~transformers.Trainer.remove_callback`]
            method.
        optimizers (`tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]`, *optional*, defaults to `(None, None)`):
            A tuple containing the optimizer and the scheduler to use. Will default to an instance of [`AdamW`] on your
            model and a scheduler given by [`get_linear_schedule_with_warmup`] controlled by `args`.
        peft_config ([`~peft.PeftConfig`], *optional*, defaults to `None`):
            PEFT configuration used to wrap the model. If `None`, the model is not wrapped.
    """

    def __init__(
            self,
            model: Union[str, PreTrainedModel],
            model_id: str,
            processor,
            reward_funcs: Union[RewardFunc, list[RewardFunc]],
            metric_funcs: Union[MetricFunc, list[MetricFunc]],
            model_init_kwargs=None,
            model_handler=None,
            model_hyper_parameters=None,
            args: GRPOConfig = None,
            train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
            eval_dataset: Optional[
                Union[Dataset, IterableDataset, dict[str, Union[Dataset, IterableDataset]]]
            ] = None,
            reward_processing_classes: Optional[
                Union[PreTrainedTokenizerBase, list[PreTrainedTokenizerBase]]
            ] = None,
            callbacks: Optional[list[TrainerCallback]] = None,
            optimizers: tuple[
                Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LambdaLR]
            ] = (None, None),
            peft_config: Optional["PeftConfig"] = None,
            max_pixels: Optional[int] = 12845056,
            min_pixels: Optional[int] = 3136,
            attn_implementation: str = "flash_attention_2",
            prompt_cls: Dict = dict(type="Qwen3Prompts"),
            prompt_task="temporal_grounding",
            **kwargs
    ):

        # Args
        if args is None:
            model_name = model if isinstance(model, str) else model.config._name_or_path
            model_name = model_name.split("/")[-1]
            args = GRPOConfig(f"{model_name}-GRPO")

        # Models
        logger.info(f"fix_vit: {args.fix_vit}", )
        fix_vit_enabled = hasattr(args, "fix_vit") and args.fix_vit

        if fix_vit_enabled and not isinstance(model, PeftModel):
            if hasattr(model, "visual"):
                logger.info(
                    "[INFO] fix_vit=True and LoRA not used. Applying ViT freezing logic..."
                )
                model.visual.requires_grad_(False)
                if hasattr(model.visual, "merger"):
                    logger.info("merger exists")
                    model.visual.merger.requires_grad_(True)
            else:
                logger.info(
                    "[WARNING] fix_vit=True but model.visual attribute not found. No freezing applied."
                )
        elif fix_vit_enabled and isinstance(model, PeftModel):
            logger.info("[INFO] fix_vit=True ignored because LoRA/PEFT is enabled.")
        elif hasattr(args, "fix_vit"):  # fix_vit exists but is False
            logger.info("[INFO] fix_vit=False. ViT freezing logic skipped.")

        # Reference model
        self.beta = args.beta
        self.use_grpo = args.use_grpo
        logger.info(f"self.use_grpo: {self.use_grpo}", )

        if self.beta == 0.0:
            self.ref_model = None
        elif is_deepspeed_zero3_enabled():
            self.ref_model, _ = load_hosted_model(
                model_path=model_id,
                model_handler=model_handler,
                model_hyper_parameters=model_hyper_parameters,
                dtype=model_init_kwargs.get("dtype", "auto"),
                model_init_kwargs=model_init_kwargs,
                peft_config=None
            )
        elif peft_config is None:
            self.ref_model = create_reference_model(model)
        else:
            self.ref_model = None

        # Processing class
        pad_token_id = processor.tokenizer.pad_token_id
        processor.image_processor.max_pixels = max_pixels
        processor.image_processor.min_pixels = min_pixels
        self.processor = processor

        sig = inspect.signature(model.forward)
        print(set(sig.parameters.keys()))

        # Reward functions
        if not isinstance(reward_funcs, list):
            reward_funcs = [reward_funcs]
        for i, reward_func in enumerate(reward_funcs):
            if isinstance(reward_func, str):
                reward_funcs[i] = AutoModelForSequenceClassification.from_pretrained(
                    reward_func, num_labels=1, **model_init_kwargs
                )
            elif isinstance(reward_func, dict):
                reward_funcs[i] = build_function(reward_func)
        self.reward_funcs = reward_funcs
        self.metric_funcs = metric_funcs

        self.prompt_cls = build_prompt(prompt_cls)
        self.decode_func = getattr(self.prompt_cls, f"decode_for_{prompt_task}")

        # Reward processing class
        if reward_processing_classes is None:
            reward_processing_classes = [None] * len(reward_funcs)
        elif not isinstance(reward_processing_classes, list):
            reward_processing_classes = [reward_processing_classes]
        else:
            if len(reward_processing_classes) != len(reward_funcs):
                raise ValueError(
                    "The number of reward processing classes must match the number of reward functions."
                )

        for i, (reward_processing_class, reward_func) in enumerate(
                zip(reward_processing_classes, reward_funcs)
        ):
            if isinstance(reward_func, PreTrainedModel):
                if reward_processing_class is None:
                    reward_processing_class = AutoTokenizer.from_pretrained(
                        reward_func.config._name_or_path
                    )
                if reward_processing_class.pad_token_id is None:
                    reward_processing_class.pad_token = (
                        reward_processing_class.eos_token
                    )
                reward_func.config.pad_token_id = reward_processing_class.pad_token_id
                reward_processing_classes[i] = reward_processing_class
        self.reward_processing_classes = reward_processing_classes

        # Data collator
        def data_collator(features):
            data = {}
            for key in features[0].keys():
                data[key] = [feat[key] for feat in features]
            return data

        # Training arguments
        self.logit_generation_batch_size = args.logit_generation_batch_size
        self.max_prompt_length = args.max_prompt_length
        self.max_completion_length = (
            args.max_completion_length
        )  # = |o_i| in the GRPO paper
        logger.info(f"max_completion_length: {self.max_completion_length}")
        self.num_generations = args.num_generations  # = G in the GRPO paper
        self.temperature = args.temperature
        self.generation_config = GenerationConfig(
            max_new_tokens=self.max_completion_length,
            do_sample=True,
            temperature=self.temperature,  # HACK
            num_return_sequences=self.num_generations,
            pad_token_id=pad_token_id,
        )
        logger.info(self.generation_config)
        logger.info(f"self.generation_config.transformers_version: {self.generation_config.transformers_version}")
        args.epsilon = 0.2
        args.epsilon_high = None
        self.epsilon_low = args.epsilon
        self.epsilon_high = (
            args.epsilon_high if args.epsilon_high is not None else args.epsilon
        )
        logger.info(f"self.beta: {self.beta}", )
        logger.info(f"self.temperature: {self.temperature}", )

        # The trainer estimates the number of FLOPs (floating-point operations) using the number of elements in the
        # input tensor associated with the key "input_ids". However, in GRPO, the sampled data does not include the
        # "input_ids" key. Instead, the available keys is "prompt". As a result, the trainer issues the warning:
        # "Could not estimate the number of tokens of the input, floating-point operations will not be computed." To
        # suppress this warning, we set the "estimate_tokens" key in the model's "warnings_issued" dictionary to True.
        # This acts as a flag to indicate that the warning has already been issued.
        model.warnings_issued["estimate_tokens"] = True

        # Initialize the metrics
        self._metrics = defaultdict(list)

        # fix bug: None of the inputs have requires_grad=True. Gradients will be None.
        model.enable_input_require_grads()

        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processor,
            callbacks=callbacks,
            optimizers=optimizers,
        )

        # Gradient accumulation requires scaled loss. Normally, loss scaling in the parent class depends on whether the
        # model accepts loss-related kwargs. Since we compute our own loss, this check is irrelevant. We set
        # self.model_accepts_loss_kwargs to False to enable scaling.
        self.model_accepts_loss_kwargs = False

        if self.ref_model is not None:
            if self.is_deepspeed_enabled:
                self.ref_model = prepare_deepspeed(self.ref_model, self.accelerator)
            else:
                self.ref_model = self.accelerator.prepare_model(
                    self.ref_model, evaluation_mode=True
                )

        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(reward_func, PreTrainedModel):
                self.reward_funcs[i] = self.accelerator.prepare_model(
                    reward_func, evaluation_mode=True
                )

    def _set_signature_columns_if_needed(self):
        # If `self.args.remove_unused_columns` is True, non-signature columns are removed.
        # By default, this method sets `self._signature_columns` to the model's expected inputs.
        # In GRPOTrainer, we preprocess data, so using the model's signature columns doesn't work.
        # Instead, we set them to the columns expected by the `training_step` method, hence the override.
        if self._signature_columns is None:
            self._signature_columns = ["prompt"]

    # Get the per-token log probabilities for the completions for the model and the reference model
    def _get_per_token_logps(
            self, model, input_ids, attention_mask, pixel_values_videos, video_grid_thw, prompt_length,
            compute_entropy=False
    ):
        logits = model(
            input_ids,
            attention_mask=attention_mask,
            pixel_values_videos=pixel_values_videos,
            video_grid_thw=video_grid_thw,
        ).logits  # (B, L, V)
        logits = logits[:, :-1, :]  # (B, L-1, V), exclude the last logit: it corresponds to the next token pred
        input_ids = input_ids[:, 1:]  # (B, L-1), exclude the first input ID since we don't have logits for it
        # Compute the log probabilities for the input tokens. Use a loop to reduce memory peak.
        per_token_logps = []
        per_token_entropy = []

        for logits_row, input_ids_row in zip(logits, input_ids):
            logZ = torch.logsumexp(logits_row, dim=-1)  # (L-1)
            token_logits = torch.gather(logits_row, 1, input_ids_row.unsqueeze(1)).squeeze(1)  # (L-1)
            token_log_prob = token_logits - logZ
            per_token_logps.append(token_log_prob)

            # entropy：分块累加 -sum p log p
            if compute_entropy:
                chunk_size = 4096
                Lm1, V = logits_row.shape
                sum_p_logp = torch.zeros(Lm1, device=logits_row.device, dtype=logits_row.dtype)

                # 广播用，避免重复分配
                logZ_col = logZ.unsqueeze(1)  # (L-1, 1)

                for start in range(0, V, chunk_size):
                    end = min(start + chunk_size, V)
                    chunk = logits_row[:, start:end]  # (L-1, chunk)
                    logp = chunk - logZ_col  # (L-1, chunk)
                    sum_p_logp += (logp.exp() * logp).sum(dim=-1)

                entropy = -sum_p_logp
                per_token_entropy.append(entropy)

        # Get rid of the prompt (-1 because of the shift done in get_per_token_logps)
        per_token_logps = torch.stack(per_token_logps)
        per_token_logps = per_token_logps[:, prompt_length - 1:]
        entropy_completion = None
        if compute_entropy:
            per_token_entropy = torch.stack(per_token_entropy)
            entropy_completion = per_token_entropy[:, prompt_length - 1:]

        return per_token_logps, entropy_completion

    def _get_per_token_logps_2(
            self, model, input_ids, attention_mask, pixel_values_videos, video_grid_thw, prompt_length, query_ids,
            query_attention_mask,
            compute_entropy=False, batch_size=1
    ):
        if batch_size is None:
            batch_size = input_ids.size(0)  # Chunk inputs into smaller batches to reduce memory peak
        all_logps = []
        all_entropies = []
        for start in range(0, input_ids.size(0), batch_size):
            input_ids_batch = input_ids[start: start + batch_size]
            attention_mask_batch = attention_mask[start: start + batch_size]
            pixel_values_videos_batch = pixel_values_videos.repeat(batch_size, 1)
            video_grid_thw_batch = video_grid_thw.repeat_interleave(batch_size, dim=0)
            query_ids_batch = query_ids.repeat_interleave(batch_size, dim=0) if query_ids is not None else None
            query_attention_mask_batch = query_attention_mask.repeat_interleave(batch_size,
                                                                                dim=0) if query_attention_mask is not None else None
            logits = model(
                input_ids_batch,
                query_ids=query_ids_batch,
                query_attention_mask=query_attention_mask_batch,
                attention_mask=attention_mask_batch,
                pixel_values_videos=pixel_values_videos_batch,
                video_grid_thw=video_grid_thw_batch,
                use_cache=False
            ).logits  # (B, L, V)
            # Exclude the last value: it corresponds to the next token pred
            logits = logits[:, :-1, :]  # (B, L-1, H)
            # Only keep the last logits_to_keep. For model that support logits_to_keep, this is a no-op.
            logits = logits[:, prompt_length - 1:, :]  # (B, logits_to_keep, H)
            # Divide logits by sampling temperature.
            # See https://huggingface.co/blog/the_n_implementation_details_of_rlhf_with_ppo#policy-training-implementation-details
            logits = logits / self.temperature
            completion_ids = input_ids_batch[:, prompt_length:]
            logps = selective_log_softmax(logits, completion_ids)  # compute logprobs
            all_logps.append(logps)

            if compute_entropy:
                with torch.no_grad():
                    entropies = entropy_from_logits(logits)
                all_entropies.append(entropies)

        logps = torch.cat(all_logps, dim=0)
        entropies = torch.cat(all_entropies, dim=0) if compute_entropy else None
        return logps, entropies

    # Trainer "prepares" the inputs before calling `compute_loss`. It converts to tensor and move to device.
    # Since we preprocess the data in `compute_loss`, we need to override this method to skip this step.
    def _prepare_inputs(
            self, inputs: dict[str, Union[torch.Tensor, Any]]
    ) -> dict[str, Union[torch.Tensor, Any]]:
        return inputs

    def compute_loss(
            self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        if return_outputs:
            raise ValueError("The GRPOTrainer does not support returning outputs")

        image_inputs = None
        video_inputs = inputs["video_inputs"]
        video_kwargs = inputs["video_kwargs"]
        processor_video_kwargs = normalize_video_kwargs_for_processor(video_kwargs[0])
        prompts = inputs["prompt"]
        queries = inputs["problem"]

        prompts_text = [
            self.processor.apply_chat_template(
                prompt, tokenize=False, add_generation_prompt=True
            )
            for prompt in prompts
        ]

        if isinstance(video_inputs[0], list):
            # qwen3 returns video_metadatas along with video_inputs
            if video_inputs is not None:
                video_inputs, video_metadatas = zip(*video_inputs)
                video_inputs, video_metadatas = list(video_inputs), list(video_metadatas)
            else:
                video_metadatas = None
            prompt_inputs = self.processor(
                text=prompts_text,
                images=image_inputs,
                videos=video_inputs,
                video_metadata=video_metadatas,
                return_tensors="pt",
                add_special_tokens=False,
                do_resize=False,
                **processor_video_kwargs)
            prompt_inputs = expand_video_grid_thw_inplace(prompt_inputs)
        else:
            prompt_inputs = self.processor(
                text=prompts_text,
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
                padding_side="left",
                add_special_tokens=False,
                **processor_video_kwargs
            )
        prompt_inputs = super()._prepare_inputs(prompt_inputs)

        if queries is not None:
            q_tok = self.processor.tokenizer(
                queries,
                padding=True,
                truncation=True,
                return_tensors="pt",
                add_special_tokens=True,
            )
            q_tok = super()._prepare_inputs(q_tok)
            prompt_inputs["query_ids"] = q_tok["input_ids"]
            prompt_inputs["query_attention_mask"] = q_tok.get("attention_mask", None)

        # print(prompt_inputs["input_ids"].shape)
        prompt_ids, prompt_mask = (
            prompt_inputs["input_ids"],
            prompt_inputs["attention_mask"],
        )

        # Generate completions
        with unwrap_model_for_generation(model, self.accelerator) as unwrapped_model:
            unwrapped_model.eval()
            with torch.no_grad():
                prompt_completion_ids = unwrapped_model.generate(
                    **prompt_inputs,
                    generation_config=self.generation_config,
                    use_model_defaults=False,
                )
            unwrapped_model.train()

            prompt_length = prompt_ids.size(1)
            # prompt_ids = prompt_completion_ids[:, :prompt_length]
            completion_ids = prompt_completion_ids[:, prompt_length:]
            # print(prompt_completion_ids.size(1))
            prompt_mask = prompt_mask.repeat_interleave(self.num_generations, dim=0)

        # Mask everything after the first EOS token
        is_eos = completion_ids == self.processor.eos_token_id
        device = self.accelerator.device
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
        completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()

        # Concatenate prompt_mask with completion_mask for logit computation
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)  # (B*G, P+C)
        # pixel_values_videos = prompt_inputs["pixel_values_videos"].repeat(self.num_generations, 1)
        # video_grid_thw = prompt_inputs["video_grid_thw"].repeat_interleave(self.num_generations, dim=0)

        del video_inputs, prompt_ids, prompt_mask

        per_token_logps, entropy_completion = self._get_per_token_logps_2(
            model,
            prompt_completion_ids,
            attention_mask,
            prompt_inputs["pixel_values_videos"],
            prompt_inputs["video_grid_thw"],
            prompt_length,
            query_ids=prompt_inputs["query_ids"],
            query_attention_mask=prompt_inputs["query_attention_mask"],
            batch_size=self.logit_generation_batch_size
        )

        if self.beta != 0.0:
            with torch.inference_mode():
                if self.ref_model is not None:
                    ref_per_token_logps, _ = self._get_per_token_logps_2(
                        self.ref_model,
                        prompt_completion_ids,
                        attention_mask,
                        prompt_inputs["pixel_values_videos"],
                        prompt_inputs["video_grid_thw"],
                        prompt_length,
                        query_ids=prompt_inputs["query_ids"],
                        query_attention_mask=prompt_inputs["query_attention_mask"],
                        batch_size=self.logit_generation_batch_size
                    )
                else:
                    with self.accelerator.unwrap_model(model).disable_adapter():
                        ref_per_token_logps, _ = self._get_per_token_logps_2(
                            model,
                            prompt_completion_ids,
                            attention_mask,
                            prompt_inputs["pixel_values_videos"],
                            prompt_inputs["video_grid_thw"],
                            prompt_length,
                            query_ids=prompt_inputs["query_ids"],
                            query_attention_mask=prompt_inputs["query_attention_mask"],
                            batch_size=self.logit_generation_batch_size
                        )

            # Compute the KL divergence between the model and the reference model
            per_token_kl = (
                    torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1)

        # Decode the generated completions
        completions = self.processor.batch_decode(completion_ids, skip_special_tokens=True)
        if is_conversational(inputs):
            completions = [
                [{"role": "assistant", "content": completion}]
                for completion in completions
            ]

        # Compute the rewards
        prompts = [prompt for prompt in prompts for _ in range(self.num_generations)]
        rewards_per_func = torch.zeros(len(prompts), len(self.reward_funcs), device=device)
        predictions = [self.decode_func(comp) for comp in completions]
        for i, (reward_func, reward_processing_class) in enumerate(
                zip(self.reward_funcs, self.reward_processing_classes)
        ):
            if isinstance(reward_func, PreTrainedModel):
                if is_conversational(inputs):
                    messages = [{"messages": p + c} for p, c in zip(prompts, completions)]
                    texts = [apply_chat_template(x, reward_processing_class)["text"] for x in messages]
                else:
                    texts = [p + c for p, c in zip(prompts, completions)]
                reward_inputs = reward_processing_class(
                    texts,
                    return_tensors="pt",
                    padding=True,
                    padding_side="right",
                    add_special_tokens=False,
                )
                reward_inputs = super()._prepare_inputs(reward_inputs)
                with torch.inference_mode():
                    rewards_per_func[:, i] = reward_func(**reward_inputs).logits[:, 0]  # Shape (B*G,)
            else:
                reward_kwargs = {
                    key: []
                    for key in inputs.keys()
                    if key not in ["prompt", "completion", "video_inputs", "video_kwargs", "image_inputs"]
                }
                for key in reward_kwargs:
                    for example in inputs[key]:
                        reward_kwargs[key].extend([example] * self.num_generations)
                output_reward_func = reward_func(predictions=predictions, prompts=prompts, completions=completions,
                                                 **reward_kwargs)
                rewards_per_func[:, i] = torch.tensor(output_reward_func, dtype=torch.float32, device=device)
                solution = reward_kwargs['solution']

        # Sum the rewards from all reward functions
        rewards = rewards_per_func.sum(dim=1)
        # Compute grouped-wise rewards
        mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)
        std_grouped_rewards = rewards.view(-1, self.num_generations).std(dim=1)
        # Normalize the rewards to compute the advantages
        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        std_grouped_rewards = std_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        advantages = (rewards - mean_grouped_rewards) / (std_grouped_rewards + 1e-4)
        if self.use_grpo:
            # x - x.detach() allows for preserving gradients from x
            per_token_loss = torch.exp(per_token_logps - per_token_logps.detach()) * advantages.unsqueeze(1)
            if self.beta != 0.0:
                per_token_loss = -(per_token_loss - self.beta * per_token_kl)
            else:
                per_token_loss = -(per_token_loss)
            loss = ((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()

        else:
            coef_1 = torch.exp(per_token_logps - per_token_logps.detach())
            coef_2 = torch.clamp(coef_1, 1 - self.epsilon_low, 1 + self.epsilon_high)
            per_token_loss1 = coef_1 * advantages.unsqueeze(1)
            per_token_loss2 = coef_2 * advantages.unsqueeze(1)
            per_token_loss = -torch.min(per_token_loss1, per_token_loss2)
            if self.beta != 0.0:
                per_token_loss = per_token_loss + self.beta * per_token_kl
            loss = (per_token_loss * completion_mask).sum() / completion_mask.sum()

        # Log the metrics
        logger.info(
            f"---------- loss: {loss:.4f} --- mem alloc: {torch.cuda.max_memory_allocated() / 1024 ** 3:.2f} GB --- mem reserved: {torch.cuda.max_memory_reserved() / 1024 ** 3:.2f} GB ----------")
        for i in range(len(predictions)):
            logger.info(f"Content: {completions[i]}")
            logger.info(f"Pred second: {predictions[i][0]} {predictions[i][1]}")
            logger.info(f"GT second: {solution[i][0]} {solution[i][1]}")
            logger.info(f"Reward: {[r.item() for r in rewards_per_func[i, :]]}")
        completion_length = (self.accelerator.gather_for_metrics(completion_mask.sum(1)).float().mean().item())
        self._metrics["completion_length"].append(completion_length)
        reward_per_func = self.accelerator.gather_for_metrics(rewards_per_func).mean(0)
        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(reward_func, PreTrainedModel):
                reward_func_name = reward_func.config._name_or_path.split("/")[-1]
            else:
                reward_func_name = reward_func.__class__.__name__
            self._metrics[f"rewards/{reward_func_name}"].append(reward_per_func[i].item())
        self._metrics["reward"].append(self.accelerator.gather_for_metrics(rewards).mean().item())
        self._metrics["reward_std"].append(self.accelerator.gather_for_metrics(std_grouped_rewards).mean().item())
        if self.beta != 0.0:
            mean_kl = ((per_token_kl * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()
            self._metrics["kl"].append(self.accelerator.gather_for_metrics(mean_kl).mean().item())
        # completion_lengths = completion_mask.sum(dim=1).clamp(min=1)
        # mean_completion_entropy = (entropy_completion * completion_mask).sum(dim=1) / completion_lengths
        # # calc mean entropy per batch
        # batch_mean_entropy = mean_completion_entropy.mean()
        # self._metrics["generation_entropy"].append(
        #     self.accelerator.gather_for_metrics(batch_mean_entropy).mean().item())

        # if torch.cuda.is_available():
        #     torch.cuda.empty_cache()
        #     get_accelerator().empty_cache()

        return loss

    def log(self, logs: dict[str, float], start_time: Optional[float] = None) -> None:
        metrics = {
            key: sum(val) / len(val) for key, val in self._metrics.items()
        }  # average the metrics
        logs = {**logs, **metrics}
        if version.parse(transformers.__version__) >= version.parse("4.47.0.dev0"):
            super().log(logs, start_time)
        else:  # transformers<=4.46
            super().log(logs)
        self._metrics.clear()


def expand_video_grid_thw_inplace(prompt_inputs):
    # prompt_inputs: BatchFeature 或 dict
    data = prompt_inputs.data if hasattr(prompt_inputs, "data") else prompt_inputs

    vgt = data["video_grid_thw"]  # 可能是 tensor / list
    if isinstance(vgt, list):
        vgt = torch.tensor(vgt, dtype=torch.long)
    if vgt.ndim == 1:
        vgt = vgt.unsqueeze(0)  # [3] -> [1,3]
    vgt = vgt.to(dtype=torch.long)

    # vgt 必须是 [1,3]，里面是 [T,H,W]
    assert vgt.shape[0] == 1 and vgt.shape[1] == 3, f"unexpected video_grid_thw shape: {vgt.shape}"

    T, H, W = map(int, vgt[0].tolist())

    expanded = torch.tensor([[1, H, W]] * T, dtype=torch.long)

    # 放到和 input_ids 同设备（generate 里会用到）
    if "input_ids" in data and isinstance(data["input_ids"], torch.Tensor):
        expanded = expanded.to(data["input_ids"].device)

    data["video_grid_thw"] = expanded
    return prompt_inputs
