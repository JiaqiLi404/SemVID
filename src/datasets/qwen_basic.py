import qwen_vl_utils
from .base import BaseDataset
from .builder import DATASETS, build_prompt
import copy


def _drop_none_values(d):
    return {k: v for k, v in d.items() if v is not None}


def normalize_video_kwargs_for_processor(video_kwargs):
    if not isinstance(video_kwargs, dict):
        return {}

    normalized = dict(video_kwargs)
    fps = normalized.get("fps")
    if isinstance(fps, (list, tuple)):
        if len(fps) == 0:
            normalized.pop("fps")
        else:
            normalized["fps"] = fps[0]
    return normalized


@DATASETS.register_module()
class BasicDatasetForQwen(BaseDataset):
    def __init__(
            self,
            dataset_name,
            train_data_path,
            eval_data_path,
            train_video_folder,
            eval_video_folder,
            subtitle_folder=None,
            video_name_only=False,
            video_name_postfix="",
            is_shuffle=True,
            video_pipeline=None,
            sample_amount=None,
            add_choice_indices=False,
            logger=None,
            is_train=True,
            seed=42,

            qwen_version=3,
            total_pixels_token_length=16384,
            min_pixels_token_length=16,
            max_frames=768,
            prompt_cls=None,
            prompt_task="temporal_grounding",
            skip_loading_video=False,
            **kwargs
    ):
        super().__init__(
            dataset_name=dataset_name,
            train_data_path=train_data_path,
            eval_data_path=eval_data_path,
            train_video_folder=train_video_folder,
            eval_video_folder=eval_video_folder,
            subtitle_folder=subtitle_folder,
            video_name_only=video_name_only,
            video_name_postfix=video_name_postfix,
            is_shuffle=is_shuffle,
            video_pipeline=video_pipeline,
            sample_amount=sample_amount,
            add_choice_indices=add_choice_indices,
            logger=logger,
            is_train=is_train,
            seed=seed,
            **kwargs
        )
        if prompt_cls is None:
            prompt_cls = dict(type="Qwen3Prompts")
        self.prompt_cls = build_prompt(prompt_cls)
        self.prompt_func = getattr(self.prompt_cls, f"for_{prompt_task}")

        if qwen_version == 3:
            self.patch_size = 16
        else:
            self.patch_size = 14
        self.qwen_version = qwen_version
        self.total_pixels = total_pixels_token_length * self.patch_size * self.patch_size * 4
        self.min_pixels = min_pixels_token_length * self.patch_size * self.patch_size * 4
        self.prompt_task = prompt_task
        self.skip_loading_video = skip_loading_video
        self.max_frames = max_frames

        self.dataset = self.dataset.with_transform(self._transform)

    def _process_one(self, data: dict) -> dict:
        # with_transform 传进来的 data 是“原始字段dict”，这里可以安全 copy，避免副作用
        data = copy.deepcopy(data)

        prompts = [
            {
                "role": "user",
                "content": [
                    _drop_none_values({
                        "type": "video",
                        "video": data["video_path"],
                        "video_start": data.get("video_start"),
                        "video_end": data.get("video_end"),
                        "total_pixels": self.total_pixels,
                        "min_pixels": self.min_pixels,
                        "max_frames": self.max_frames,
                    }),
                    {"type": "text", "text": self.prompt_func(event=data["problem"], choices=data["choices"])},
                ],
            },
        ]

        if self.skip_loading_video:
            image_inputs = video_inputs = video_kwargs = None
        else:
            image_inputs, video_inputs, video_kwargs =process_vision_info_qwen(prompts,self.qwen_version,self.patch_size)

        data.update({
            "image_inputs": image_inputs[0] if image_inputs is not None else None,
            "video_inputs": video_inputs[0] if video_inputs is not None else None,
            "video_kwargs": video_kwargs[0] if video_inputs is not None else None,
            "prompt": prompts
        })

        # 如果你确实要在这里跑 mmengine pipeline（比如抽帧/增强），放这里最合适：
        if self.video_pipeline is not None:
            data = self.video_pipeline(data)

        return data

    def _transform(self, examples: dict) -> dict:
        """
        HF with_transform 可能在以下两种情况下调用：
        - 单样本索引：examples 是 dict(str -> value)
        - 切片/批量索引：examples 是 dict(str -> list[value])
        我们同时兼容。
        """
        # 判断是否 batch
        any_key = next(iter(examples.keys()))
        is_batched = isinstance(examples[any_key], list)

        if not is_batched:
            return self._process_one(examples)

        # batched：逐个处理，再拼回 dict of lists
        outs = []
        n = len(examples[any_key])
        for i in range(n):
            one = {k: v[i] for k, v in examples.items()}
            outs.append(self._process_one(one))

        merged = {k: [o.get(k) for o in outs] for k in outs[0].keys()}
        return merged

    # ---- proxy (optional; BaseDataset already provides) ----
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]


def process_vision_info_qwen(prompts_batch, qwen_version=3, patch_size=16):
    if not isinstance(prompts_batch[0], list):
        prompts_batch = [prompts_batch]

    vision_batch = []
    for prompts in prompts_batch:
        # prompts[0]["content"] 中取 video block
        vb = None
        for c in prompts[0]["content"]:
            if isinstance(c, dict) and c.get("type") == "video":
                vb = c
                break
        if vb is None:
            raise ValueError("Cannot find video block in prompts[0]['content'].")
        # print(prompts)

        messages = [
            {
                "role": "user",
                "content": [
                    _drop_none_values({
                        "type": "video",
                        "video": vb["video"],
                        "total_pixels": vb.get("total_pixels"),
                        "min_pixels": vb.get("min_pixels"),
                        "video_start": vb.get("video_start"),
                        "video_end": vb.get("video_end"),
                    })
                ],
            }
        ]
        vision_batch.append(messages)

    B = len(vision_batch)
    qv = qwen_version

    # 2) 调用 process_vision_info（按你贴的源码语义）
    if qv == 2 or qv == 2.0:
        image_inputs, video_inputs = qwen_vl_utils.process_vision_info(vision_batch)
        # dataset 对 qwen2: video_kwargs = {}
        per_sample_kwargs = [{} for _ in range(B)]
    elif qv == 2.5:
        # dataset 对 qwen2.5: return_video_kwargs=True, 不传 image_patch_size/return_video_metadata
        image_inputs, video_inputs, video_kwargs = qwen_vl_utils.process_vision_info(
            vision_batch,
            return_video_kwargs=True,
        )
        # Keep fps per sample. Newer Transformers validates fps as a scalar,
        # not as a one-element list.
        base = {"do_sample_frames": video_kwargs.get("do_sample_frames", False)}
        fps_list = video_kwargs.get("fps", None)  # 源码里是 list，长度等于 batch 内 video 数
        if fps_list is None:
            per_sample_kwargs = [dict(base) for _ in range(B)]
        else:
            if len(fps_list) != B:
                raise ValueError(f"video_kwargs['fps'] length {len(fps_list)} != batch {B}")
            per_sample_kwargs = [dict(base, fps=fps_list[i]) for i in range(B)]
    elif qv == 3 or qv == 3.0:
        image_inputs, video_inputs, video_kwargs = qwen_vl_utils.process_vision_info(
            vision_batch,
            image_patch_size=patch_size,
            return_video_kwargs=True,
            return_video_metadata=True,
        )
        base = {"do_sample_frames": video_kwargs.get("do_sample_frames", False)}
        per_sample_kwargs = [dict(base) for _ in range(B)]
    else:
        raise ValueError(f"Unsupported qwen_version: {qwen_version}")

    # 3) 校验长度（每个 conversation 只有一个 video，所以 inputs 应该与 B 对齐）
    if image_inputs is not None and len(image_inputs) != B:
        raise ValueError(f"len(image_inputs)={len(image_inputs)} != batch={B}")
    if video_inputs is not None and len(video_inputs) != B:
        raise ValueError(f"len(video_inputs)={len(video_inputs)} != batch={B}")

    return image_inputs, video_inputs, per_sample_kwargs
