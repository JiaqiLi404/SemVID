import json
import os
import traceback
from typing import List, Optional, Any, Dict, Type, TypeVar, Tuple

import torch
from tqdm import tqdm
from transformers import AutoProcessor, GenerationConfig
from ..builder import EVALUATORS
from ..utils import setup_root_logging
from ...datasets.llava_basic import process_vision_info_llava
from ...utils.model_utils import load_hosted_model


@EVALUATORS.register_module()
class LLaVATransformersEvaluator:
    """
    KnownIssue:
    #############################################################################################
    ## There are some remaining issue maybe correlated to the position embedding or padding,   ##
    ## setting the batch_size!=1 may lead to unstable problem                                  ##
    #############################################################################################
    """

    def __init__(
            self,
            model_path: str,
            model_handler: str,
            model_hyper_parameters: dict[str, Any],
            max_new_tokens: int,
            dtype: str = "auto",
            trust_remote_code: bool = True,
            log_path: Optional[str] = None,
            pred_path: Optional[str] = None,
            fail_fast: bool = True,  # batch 出错是否直接 raise
            force_stop_thinking: bool = False,
            **kwargs
    ):
        self.fail_fast = fail_fast
        self.force_stop_thinking = force_stop_thinking
        self.pred_path = pred_path
        self.log_path = log_path

        setup_root_logging(self.log_path)

        self.model, self.processor = load_hosted_model(model_path, model_handler=model_handler, dtype=dtype,
                                                       backend="vllm",
                                                       model_hyper_parameters=model_hyper_parameters)
        self.model.eval()

        if self.model.config.text_config.pad_token_id is None:
            self.model.config.text_config.pad_token_id = self.processor.tokenizer.pad_token_id

        # generation config
        self.generation_config = GenerationConfig(
            max_new_tokens=int(max_new_tokens),
            do_sample=False,  # temperature=0
            temperature=0.0,
            top_p=1.0,
            # 你可以按需加：repetition_penalty、num_beams 等
        )

        if self.pred_path is not None:
            os.makedirs(os.path.dirname(self.pred_path), exist_ok=True)

        import qwen_vl_utils
        self.qwen_vl_utils = qwen_vl_utils

    @staticmethod
    def build_driver_payload(item: Dict[str, Any], idx: int) -> Dict[str, Any]:
        """driver 端只做轻量打包，不加载 tokenizer，不渲染 prompt string。"""
        sample_id = item.get("id", item.get("sample_id", idx))

        payload = {
            "idx": idx,
            "id": str(sample_id),
            "prompt": item["prompt"],
            "query": item["problem"],
            "target": item["solution"],
            "video_start": item.get("video_start", None),
            "video_end": item.get("video_end", None),
            "durations": item.get("durations", None),
            "type": item.get("type", None),
            "subtitles": item.get("subtitles", None),
            "media_mode": item.get("media_mode", None),
            "video_decode_kwargs": item.get("video_decode_kwargs", None),
        }
        return payload

    def _append_preds(self, records: List[Dict[str, Any]]):
        with open(self.pred_path, "a", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
            f.flush()

    # ----------------- helpers -----------------
    def _move_inputs_to_model_device(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """device_map=auto 时，模型可能分布在多卡；inputs 放到 model.device 通常可行。"""
        dev = getattr(self.model, "device", None)
        if dev is None:
            # 某些情况下 model.device 不可靠；退化到 cuda:0 / cpu
            dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        for k, v in list(inputs.items()):
            if torch.is_tensor(v):
                inputs[k] = v.to(dev)
        return inputs

    def _decode(self, gen_ids: torch.LongTensor) -> List[str]:
        """gen_ids: (B, T)"""
        return self.processor.batch_decode(gen_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)

    @staticmethod
    def _safe_get_prompt_len(input_ids: Optional[torch.Tensor]) -> int:
        if input_ids is None:
            return 0
        return int(input_ids.shape[-1])

    def generate_requests(self, payloads: List[Dict[str, Any]], batch_size: int = 1) -> Dict[str, Any]:
        """
        返回 dict，包含：
          - "pred_path": 本 worker 的预测文件路径
          - "log_path": 本 worker 的日志文件路径
          - "num_ok": 成功条数
          - "num_err": 失败 batch 数
        """
        num_ok = 0
        num_err = 0

        for st in tqdm(range(0, len(payloads), batch_size)):
            batch = payloads[st: st + batch_size]
            bsz = len(batch)
            try:
                prompts_batch = [p["prompt"] for p in batch]
                queries_batch = [p["query"] for p in batch]
                subtitles_batch = [p["subtitles"] for p in batch]
                clip_starts = [p["video_start"] for p in batch]
                clip_ends = [p["video_end"] for p in batch]
                duration_batch = [p["durations"] for p in batch]
                media_modes = [p["media_mode"] for p in batch]
                video_decode_kwargs = [p["video_decode_kwargs"] for p in batch]
                prompts_batch, image_inputs, video_inputs, video_kwargs, frames_list, frame_ts_list, sampling_fps_list \
                    = process_vision_info_llava(prompts_batch,
                                                subtitles=subtitles_batch,
                                                clip_starts=clip_starts,
                                                clip_ends=clip_ends,
                                                durations=duration_batch,
                                                media_modes=media_modes,
                                                video_decode_kwargs=video_decode_kwargs)

                prompts_text = []
                for prompt in prompts_batch:
                    prompts_str = self.processor.apply_chat_template(
                        prompt,
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                    if self.force_stop_thinking:
                        prompts_str += "</think>"
                    prompts_text.append(prompts_str)

                prompt_inputs = self.processor(
                    text=prompts_text,
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    return_tensors="pt",
                    **video_kwargs[0])
                prompt_inputs = self._move_inputs_to_model_device(prompt_inputs)

                q_tok = self.processor.tokenizer(
                    queries_batch,
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                    add_special_tokens=True,
                )

                q_tok = self._move_inputs_to_model_device(q_tok)
                prompt_inputs["query_ids"] = q_tok["input_ids"]
                prompt_inputs["query_attention_mask"] = q_tok.get("attention_mask", None)

                gen_ids = self.model.generate(
                    **prompt_inputs,
                    generation_config=self.generation_config,
                    use_model_defaults=False,
                )
                decoded = self._decode(gen_ids)

                input_seq_len = prompt_inputs["input_ids"].shape[1]

                batch_records: List[Dict[str, Any]] = []
                for i, p in enumerate(batch):
                    text_full = decoded[i]
                    if input_seq_len > 0:
                        gen_only_ids = gen_ids[i, input_seq_len:]
                        pred_only = self._decode(gen_only_ids.unsqueeze(0))[0]
                    else:
                        pred_only = text_full

                    rec = {
                        "idx": p.get("idx"),
                        "id": p.get("id"),
                        "target": p.get("target"),
                        "pred": pred_only,
                        # "all": text_full,
                        "durations": p.get("durations", None),
                        "type": p.get("type", None)
                    }
                    batch_records.append(rec)

                self._append_preds(batch_records)
                num_ok += len(batch_records)

            except Exception as e:
                num_err += 1
                tb = traceback.format_exc()
                print(f"batch {st}:{st + bsz} failed: {e}\n{tb}")
                if self.fail_fast:
                    raise
        return {
            "pred_path": self.pred_path,
            "num_ok": num_ok,
            "num_err": num_err,
        }
