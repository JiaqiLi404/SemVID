import copy
from typing import Any, Dict, List, Literal, Optional, Tuple
from transformers import AutoProcessor

from src.datasets.qwen_basic import normalize_video_kwargs_for_processor
from src.models.builder import COLLATORS


def format_target_text_from_solution(solution) -> str:
    st, ed = solution
    return f"The given query happens in {float(st)} - {float(ed)} seconds."


@COLLATORS.register_module()
class QwenVideoSFTGRPOCollator:
    def __init__(
        self,
        processor_path: Any,
        qwen_version: Literal[2, 2.5, 3] = 3,
        mode: Literal["sft", "grpo"] = "sft",
        train_on_assistant_only: bool = True,
        do_resize: bool = False,
        **kwargs,
    ):
        self.processor = AutoProcessor.from_pretrained(processor_path)
        self.qwen_version = qwen_version
        self.mode = mode
        self.train_on_assistant_only = train_on_assistant_only
        self.do_resize = do_resize

    def __call__(self, features: List[Dict[str, Any]]):
        image_inputs = [ex.get("image_inputs", None) for ex in features]
        video_inputs = [ex.get("video_inputs", None) for ex in features]
        video_kwargs_list = [ex.get("video_kwargs", {}) or {} for ex in features]

        has_image = any(x is not None for x in image_inputs)
        has_video = any(v is not None for v in video_inputs)

        # ====== 关键：处理 Qwen3 的 (video, meta) ======
        video_metadatas = None
        if self.qwen_version == 3:
            if has_video:
                # video_inputs 期望是 list[(video, meta)]，其中某些可能为 None
                vids = []
                metas = []
                for v in video_inputs:
                    if v is None:
                        vids.append(None)
                        metas.append(None)
                        continue
                    if isinstance(v, (tuple, list)) and len(v) >= 2:
                        vids.append(v[0])
                        metas.append(v[1])
                    else:
                        # 如果有人只给了 video 没 meta，那就只能 None
                        vids.append(v)
                        metas.append(None)
                video_inputs = vids
                video_metadatas = metas
            else:
                video_metadatas = None
        else:
            # Qwen2/2.5：兼容“被包一层 list”的情况（单视频）
            def _unwrap_first_if_list(x):
                if isinstance(x, list) and len(x) > 0:
                    return x[0]
                return x
            video_inputs = [_unwrap_first_if_list(v) for v in video_inputs]

        # video_kwargs：按你可行代码，直接用第一条展开
        vk0 = normalize_video_kwargs_for_processor(video_kwargs_list[0])
        # 注意：不要在这里强行 pop fps，Qwen3 更推荐用 video_metadata 推 fps
        # 如果你 video_kwargs 里有 fps，也可以保留，processor 会优先用你传的

        # ====== 构造 prompt-only / full(with GT) ======
        prompt_texts = []
        full_texts = []
        target_texts = []
        for ex in features:
            tgt = format_target_text_from_solution(ex.get("solution"))
            target_texts.append(tgt)

            prompt_texts.append(
                self.processor.apply_chat_template(
                    ex["prompt"], tokenize=False, add_generation_prompt=True
                )
            )

            full_msgs = copy.deepcopy(ex["prompt"])
            full_msgs.append({"role": "assistant", "content": [{"type": "text", "text": tgt}]})
            full_texts.append(
                self.processor.apply_chat_template(
                    full_msgs, tokenize=False, add_generation_prompt=False
                )
            )

        # ====== processor 调用（对齐你给的“可行代码”风格） ======
        common_kwargs = dict(
            images=image_inputs if has_image else None,
            videos=video_inputs if has_video else None,
            return_tensors="pt",
            padding=True,
            padding_side="left",
            add_special_tokens=False,
        )

        if self.qwen_version == 3:
            common_kwargs.update(dict(
                video_metadata=video_metadatas,
                do_resize=self.do_resize,
            ))

        prompt_inputs = self.processor(text=prompt_texts, **common_kwargs, **vk0)

        if self.mode == "grpo":
            return prompt_inputs

        full_inputs = self.processor(text=full_texts, **common_kwargs, **vk0)

        # ====== labels：mask prompt + pad + video token ======
        prompt_inputs.pop("token_type_ids", None)
        full_inputs.pop("token_type_ids", None)

        input_ids = full_inputs["input_ids"]
        # print(input_ids.shape)
        attention_mask = full_inputs.get("attention_mask", None)

        labels = input_ids.clone()

        if self.train_on_assistant_only:
            prompt_lens = prompt_inputs["attention_mask"].sum(dim=1)  # 每条 prompt 的真实长度
            full_lens = full_inputs["attention_mask"].sum(dim=1)  # 每条 full 的真实长度
            seq_len = full_inputs["input_ids"].size(1)

            for i in range(len(features)):
                full_start = seq_len - full_lens[i]  # full 真实 token 在 padded 序列中的起点
                prompt_start = full_start
                prompt_end = full_start + prompt_lens[i]
                labels[i, prompt_start:prompt_end] = -100

        # mask padding by attention_mask
        if attention_mask is not None:
            labels = labels.masked_fill(attention_mask == 0, -100)

        # mask pad_token id (双保险)
        pad_id = self.processor.tokenizer.pad_token_id
        if pad_id is not None:
            labels[labels == pad_id] = -100

        # mask video token
        video_token = getattr(self.processor, "video_token", None)
        if video_token is not None:
            video_token_id = self.processor.tokenizer.convert_tokens_to_ids(video_token)
            labels[labels == video_token_id] = -100

        batch = dict(full_inputs)
        batch["labels"] = labels
        batch["target_text"] = target_texts
        batch["id"] = [ex.get("id") for ex in features]
        return batch
