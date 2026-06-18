import logging

import torch
from torch import nn
from trl import GRPOTrainer, is_conversational, apply_chat_template
from trl.extras.profiling import profiling_context, profiling_decorator
from accelerate.utils import gather

from src.datasets.builder import build_prompt
from src.models.builder import TRAINERS
from src.models.utils.callbacks import CUDAMemoryCallback

logger = logging.getLogger(__name__)


@TRAINERS.register_module()
class MyGRPOTrainer(GRPOTrainer):
    """
    KnownIssue:
    ##########################################################################################
    ## This version doesn't support multi-modal inputs currently, please don't use it       ##
    ##########################################################################################
    """

    def __init__(self, model,prompt_cls, prompt_task, *args, data_collator=None, **kwargs):
        self._user_data_collator = data_collator
        # skip unused args and call super
        super().__init__(
            model=model,
            reward_funcs=kwargs.get('reward_funcs'),
            args=kwargs.get('args'),
            train_dataset=kwargs.get('train_dataset'),
            eval_dataset=kwargs.get('eval_dataset'),
            processing_class=kwargs.get('processing_class'),
            reward_processing_classes=kwargs.get('reward_processing_classes'),
            tools=kwargs.get('tools'),
            rollout_func=kwargs.get('rollout_func'),
            peft_config=kwargs.get('peft_config'),
            callbacks=[CUDAMemoryCallback(every_n_steps=1, reset_peak=False)],
        )
        if data_collator is not None:
            self.data_collator = data_collator
        self.prompt_cls = build_prompt(prompt_cls)
        self.decode_func = getattr(self.prompt_cls, f"decode_for_{prompt_task}")

    @profiling_decorator
    def _calculate_rewards(self, inputs, prompts, completions, completion_ids_list):
        device = self.accelerator.device
        rewards_per_func = torch.zeros(len(prompts), len(self.reward_funcs), device=device)
        predictions = [self.decode_func(comp) for comp in completions]

        # Repeat all input columns (but "prompt", "completion", and "completion_ids") to match the num of generations
        keys = [key for key in inputs[0] if key not in ["prompt", "completion", "completion_ids"]]
        reward_kwargs = {key: [example[key] for example in inputs] for key in keys}

        # This allows for dynamic reward shaping based on training progress.
        reward_kwargs["trainer_state"] = self.state

        for i, (reward_func, reward_processing_class, reward_func_name) in enumerate(
                zip(self.reward_funcs, self.reward_processing_classes, self.reward_func_names, strict=True)
        ):
            with profiling_context(self, reward_func_name):
                if isinstance(reward_func, nn.Module):  # Module (no PretrainedModel) for compat with compiled models
                    if is_conversational(inputs[0]):
                        messages = [{"messages": p + c} for p, c in zip(prompts, completions, strict=True)]
                        texts = [
                            apply_chat_template(x, reward_processing_class, **self.chat_template_kwargs)["text"]
                            for x in messages
                        ]
                    else:
                        texts = [p + c for p, c in zip(prompts, completions, strict=True)]
                    reward_inputs = reward_processing_class(
                        text=texts, return_tensors="pt", padding=True, padding_side="right", add_special_tokens=False
                    )
                    reward_inputs = super()._prepare_inputs(reward_inputs)
                    with torch.inference_mode():
                        rewards_per_func[:, i] = reward_func(**reward_inputs).logits[:, 0]  # Shape (B*G,)
                else:
                    output_reward_func = reward_func(
                        predictions=predictions, prompts=prompts, completions=completions,
                        completion_ids=completion_ids_list, **reward_kwargs
                    )
                    # Convert None values to NaN
                    output_reward_func = [reward if reward is not None else torch.nan for reward in output_reward_func]

                    rewards_per_func[:, i] = torch.tensor(output_reward_func, dtype=torch.float32, device=device)

        # If all reward functions return None for a given row, issue a detailed warning
        if torch.isnan(rewards_per_func).all(dim=1).any():
            nan_row_idx = torch.isnan(rewards_per_func).all(dim=1).nonzero(as_tuple=True)[0][0]
            row_reward_kwargs = {
                key: value[nan_row_idx] for key, value in reward_kwargs.items() if key != "trainer_state"
            }
            row_reward_kwargs["prompt"] = prompts[nan_row_idx]
            row_reward_kwargs["completion"] = completions[nan_row_idx]
            logger.warning(
                f"All reward functions returned None for the following kwargs:\n{row_reward_kwargs}\n"
                "Please ensure that at least one reward function returns a valid reward."
            )

        # Gather the reward per function: this part is crucial, because the rewards are normalized per group and the
        # completions may be distributed across processes
        rewards_per_func = gather(rewards_per_func)

        logger.info(
            f"---------- mem alloc: {torch.cuda.max_memory_allocated() / 1024 ** 3:.2f} GB --- mem reserved: {torch.cuda.max_memory_reserved() / 1024 ** 3:.2f} GB ----------")
        for i in range(len(predictions)):
            logger.info(f"Content: {completions[i]}")
            logger.info(f"Pred second: {predictions[i][0]} {predictions[i][1]}")
            logger.info(f"GT second: {reward_kwargs['solution'][i][0]} {reward_kwargs['solution'][i][1]}")
            logger.info(f"Reward: {[r.item() for r in rewards_per_func[i, :]]}")
        return rewards_per_func
