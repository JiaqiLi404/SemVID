import json
import os
import math
import hashlib
import traceback
import time
from typing import List, Optional, Any, Dict, Type, TypeVar, Tuple

import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from transformers import AutoProcessor, GenerationConfig

from analysis.evidence_metrics import (
    compute_connectivity_strength,
    compute_evidence_cross_entropy,
    compute_landing_distribution,
)
from ..builder import EVALUATORS
from ..utils import setup_root_logging
from ...datasets.qwen_basic import normalize_video_kwargs_for_processor, process_vision_info_qwen
from ...utils.model_utils import load_hosted_model


@EVALUATORS.register_module()
class QwenTransformersEvaluator:
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
            qwen_version: float = 3,
            patch_size: Optional[int] = 16,
            dtype: str = "auto",
            trust_remote_code: bool = True,
            log_path: Optional[str] = None,
            pred_path: Optional[str] = None,
            fail_fast: bool = True,  # batch 出错是否直接 raise
            force_stop_thinking: bool = False,
            do_visualize_attn_map: bool = False,
            analysis_root: str = "analysis_outputs",
            **kwargs
    ):
        self.fail_fast = fail_fast
        self.force_stop_thinking = force_stop_thinking
        self.pred_path = pred_path
        self.log_path = log_path
        self.qwen_version = float(qwen_version)
        self.patch_size = patch_size
        self.do_visualize_attn_map = do_visualize_attn_map
        self.analysis_root = analysis_root

        setup_root_logging(self.log_path)

        self.model, self.processor = load_hosted_model(model_path, model_handler=model_handler, dtype=dtype,
                                                       backend="vllm", model_hyper_parameters=model_hyper_parameters)
        self.model.eval()

        if self.model.config.text_config.pad_token_id is None:
            self.model.config.text_config.pad_token_id = self.processor.tokenizer.pad_token_id

        # generation config
        self.generation_config = GenerationConfig(
            max_new_tokens=int(max_new_tokens),
            # max_new_tokens=1,
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
            "durations": item.get("durations", None),
            "type": item.get("type", None),
            "video_path": item.get("video_path", None)
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
                image_inputs, video_inputs, video_kwargs = process_vision_info_qwen(prompts_batch,
                                                                                    self.qwen_version,
                                                                                    self.patch_size)
                processor_video_kwargs = normalize_video_kwargs_for_processor(video_kwargs[0])
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

                if self.qwen_version == 3:
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
                        padding=True,
                        padding_side="left",
                        add_special_tokens=False,
                        do_resize=False,
                        **processor_video_kwargs)
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

                if self.do_visualize_attn_map:
                    self.visualize_attn_map(prompt_inputs, batch, video_metadatas)

                start_time=time.time()
                gen_ids = self.model.generate(
                    **prompt_inputs,
                    generation_config=self.generation_config,
                    use_model_defaults=False,
                )
                decoded = self._decode(gen_ids)
                end_time=time.time()
                print(f"seq length: {prompt_inputs['input_ids'].shape[1]}  processing time: {end_time-start_time}s")

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

    def visualize_attn_map(self, prompt_inputs, batch, video_metadatas=None):
        # ----------------------------
        # Helpers
        # ----------------------------
        def _extract_first_video_grid_thw(video_grid_thw):
            if video_grid_thw is None:
                return None
            if isinstance(video_grid_thw, (list, tuple)):
                video_grid_thw = torch.tensor(video_grid_thw)
            if not torch.is_tensor(video_grid_thw):
                return None
            vg = video_grid_thw
            if vg.dim() == 1 and vg.numel() == 3:
                t, h, w = vg.tolist()
                return int(t), int(h), int(w)
            if vg.dim() == 2 and vg.size(-1) == 3:
                t, h, w = vg[0].tolist()
                return int(t), int(h), int(w)
            if vg.dim() == 3 and vg.size(-1) == 3:
                t, h, w = vg[0, 0].tolist()
                return int(t), int(h), int(w)
            return None

        def _open_video_cv2(video_path: str):
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise RuntimeError(f"无法打开视频：{video_path}")
            fps = cap.get(cv2.CAP_PROP_FPS)
            if not fps or fps <= 0:
                fps = 30.0
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            return cap, float(fps), int(total_frames)

        def _read_frame_at_indice(cap, frame_index: int):
            if torch.is_tensor(frame_index):
                frame_index = int(frame_index.item())
            if frame_index < 0:
                return None
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame_bgr = cap.read()
            if not ok or frame_bgr is None:
                return None
            return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        def _read_frame_at_second(cap, sec0: int):
            cap.set(cv2.CAP_PROP_POS_MSEC, float(sec0) * 1000.0)
            ok, frame_bgr = cap.read()
            if not ok or frame_bgr is None:
                return None
            return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        def _safe_grid_hw_from_P(P: int, llm_grid_h: int, llm_grid_w: int):
            if llm_grid_h is not None and llm_grid_w is not None and llm_grid_h * llm_grid_w == P:
                return llm_grid_h, llm_grid_w
            s = int(math.sqrt(P))
            if s > 0 and s * s == P:
                return s, s
            # fallback: make it as square as possible
            h = int(math.floor(math.sqrt(P)))
            h = max(h, 1)
            w = int(math.ceil(P / h))
            return h, w

        def _save_pi_matrix(attn_dir: str, video_base: str, pi_fp_np: np.ndarray) -> Tuple[str, bool]:
            """
            Returns (saved_path, is_baseline_saved_now)
            """
            os.makedirs(attn_dir, exist_ok=True)
            base_path = os.path.join(attn_dir, f"{video_base}_base.npy")
            if not os.path.exists(base_path):
                np.save(base_path, pi_fp_np)
                return base_path, True

            # find next X
            x = 1
            while True:
                cand = os.path.join(attn_dir, f"{video_base}_{x}.npy")
                if not os.path.exists(cand):
                    # np.save(cand, pi_fp_np)
                    return cand, False
                x += 1

        def _visualize_pi0_on_frames(
            cap,
            frames_indices: List[int],
            pi_fp: torch.Tensor,   # [F, P], in [0,1] or normalized
            llm_grid_h: int,
            llm_grid_w: int,
            output_path: str,
            vid_id: str,
        ):
            os.makedirs(output_path, exist_ok=True)
            F_, P_ = pi_fp.shape
            gh, gw = _safe_grid_hw_from_P(P_, llm_grid_h, llm_grid_w)

            # stable scale across frames
            pi_min = float(pi_fp.min().item())
            pi_max = float(pi_fp.max().item())
            denom = max(pi_max - pi_min, 1e-6)

            out_dir = os.path.join(output_path, "pi_0")
            os.makedirs(out_dir, exist_ok=True)

            for t in range(F_):
                frame_idx = frames_indices[t] if t < len(frames_indices) else frames_indices[-1]
                frame = _read_frame_at_indice(cap, frame_idx)
                if frame is None:
                    frame = _read_frame_at_second(cap, t)
                if frame is None:
                    break

                h_img, w_img = frame.shape[:2]

                heat = (pi_fp[t] - pi_min) / denom
                heat = heat.reshape(gh, gw).detach().cpu().numpy()
                heat = cv2.resize(heat, (w_img, h_img), interpolation=cv2.INTER_CUBIC)
                heat_u8 = np.clip(heat * 255.0, 0, 255).astype(np.uint8)
                heat_color = cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET)
                heat_color = cv2.cvtColor(heat_color, cv2.COLOR_BGR2RGB)

                overlay = (0.55 * frame + 0.45 * heat_color).clip(0, 255).astype(np.uint8)

                cv2.imwrite(os.path.join(out_dir, f"raw_{vid_id}_t{t+1:04d}.png"), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                cv2.imwrite(os.path.join(out_dir, f"heat_{vid_id}_t{t+1:04d}.png"), cv2.cvtColor(heat_color, cv2.COLOR_RGB2BGR))
                cv2.imwrite(os.path.join(out_dir, f"overlay_{vid_id}_t{t+1:04d}.png"), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

        # ----------------------------
        # Run model and fetch summaries
        # ----------------------------
        video_path_batch = [p["video_path"] for p in batch]
        id_batch = [p["id"] for p in batch]

        with torch.no_grad():
            out = self.model(
                **prompt_inputs,
                return_query_visual_attentions=True,
                return_vision_self_topk=True,
                vision_self_topk_k=64,
            )

        # ----------------------------
        # Derive grid sizes (llm_grid_h/w) from video_grid_thw + merge
        # ----------------------------
        merge = None
        cfg = getattr(self.model, "config", None)
        if cfg is not None and hasattr(cfg, "vision_config") and cfg.vision_config is not None:
            merge = getattr(cfg.vision_config, "spatial_merge_size", None)
        if merge is None:
            merge = 2

        thw = _extract_first_video_grid_thw(prompt_inputs.get("video_grid_thw", None))
        if thw is None:
            raise RuntimeError("video_grid_thw is missing; cannot infer F/H/W for visualization.")

        _T, H, W = thw
        llm_grid_h = int(H // int(merge))
        llm_grid_w = int(W // int(merge))

        attn_dir = os.path.join(self.analysis_root, "attention_map")
        os.makedirs(attn_dir, exist_ok=True)

        for b in range(len(video_path_batch)):
            qv_attn = out.query_visual_attentions[b]              # [L, F, P]
            v_query = out.vision_self_topk_query[b]               # [Nv]
            v_indices = out.vision_self_topk_indices[b]           # [L, Nv, K]
            v_weights = out.vision_self_topk_weights[b]           # [L, Nv, K]

            qv_attn = qv_attn.float()
            # Use last layer's injection distribution as π^(L) (Eq. 14 "induced by cross-attention")
            pi_inject_fp = qv_attn[-1]  # [F, P]
            F_, P_ = pi_inject_fp.shape

            # 1) π_0 (landing map) via sparse evidence flow (Eq. 14)
            pi0_fp = compute_landing_distribution(
                query_visual_seed=pi_inject_fp,
                retained_indices=v_query,
                attention_indices=v_indices,
                attention_weights=v_weights,
                residual_weight=0.1,
                fill_self_loop=True,
            )  # [F, P]

            # 2) Save π_0 matrix with baseline/increment rule
            video_path = video_path_batch[b]
            video_base = os.path.splitext(os.path.basename(video_path))[0]
            sample_id = str(id_batch[b])
            sample_hash = hashlib.sha1(sample_id.encode("utf-8")).hexdigest()[:12]
            analysis_key = f"{video_base}_{sample_hash}"
            saved_path, saved_as_base = _save_pi_matrix(
                attn_dir,
                analysis_key,
                pi0_fp.detach().cpu().numpy(),
            )

            # 3) Compute CS (Eq. 12, 15) always
            cs_mean, cs_layers, gamma_layers = compute_connectivity_strength(
                retained_indices=v_query.detach().cpu(),
                attention_indices=v_indices.detach().cpu(),
                attention_weights=v_weights.detach().cpu(),
                frames=F_,
                patches=P_,
            )

            # 4) Compute ER (Eq. 11) only if baseline exists (i.e. not the first time)
            er = None
            base_path = os.path.join(attn_dir, f"{analysis_key}_base.npy")
            if (not saved_as_base) and os.path.exists(base_path):
                try:
                    pi_full_np = np.load(base_path)  # baseline π_full^(1)
                    pi_full_fp = torch.from_numpy(pi_full_np).to(pi0_fp.device).float()
                    er = compute_evidence_cross_entropy(
                        full_landing_distribution=pi_full_fp,
                        pruned_landing_distribution=pi0_fp,
                        retained_indices=v_query.to(pi0_fp.device),
                    )
                except Exception:
                    er = None

            # ----------------------------
            # Visualization
            # ----------------------------
            vid_id = sample_id
            output_path = os.path.join(self.analysis_root, "samples", analysis_key)
            os.makedirs(output_path, exist_ok=True)

            # frame indices
            frames_indices = None
            if video_metadatas is not None:
                meta = video_metadatas[b]
                if "frames_indices" in meta:
                    frames_indices = meta["frames_indices"]
                    if isinstance(frames_indices, (list, tuple)) and len(frames_indices) >= 2 * F_:
                        frames_indices = [
                            int(round((frames_indices[2 * i] + frames_indices[2 * i + 1]) / 2))
                            for i in range(F_)
                        ]

            cap, fps, total_frames = _open_video_cv2(video_path)
            if frames_indices is None:
                if total_frames <= 0:
                    frames_indices = list(range(F_))
                else:
                    xs = np.linspace(0, max(total_frames - 1, 0), num=F_)
                    frames_indices = [int(round(x)) for x in xs]

            # dump meta
            with open(os.path.join(output_path, f"meta.txt"), "w") as f:
                json.dump(batch[b], f, indent=4)

            # visualize π_0
            _visualize_pi0_on_frames(
                cap=cap,
                frames_indices=frames_indices,
                pi_fp=pi0_fp,
                llm_grid_h=llm_grid_h,
                llm_grid_w=llm_grid_w,
                output_path=output_path,
                vid_id=vid_id,
            )

            cap.release()

            # metrics json
            metrics = {
                "video_base": video_base,
                "sample_id": sample_id,
                "analysis_key": analysis_key,
                "pi0_saved_path": saved_path,
                "ER": er,
                "CS_mean": cs_mean,
                "CS_layers": cs_layers,
                "Gamma_layers": gamma_layers,
            }
            with open(os.path.join(output_path, "attn_metrics.json"), "w") as f:
                json.dump(metrics, f, indent=2)


@EVALUATORS.register_module()
class QwenVLLMEvaluator:
    """
    KnownIssue:
    ##############################################################################################################################
    ## This Evaluator only support official models, doesn't support customized model handlers, and extra inputs for generation  ##
    ##############################################################################################################################
    """

    def __init__(
            self,
            model_path: str,
            max_new_tokens: int,
            qwen_version: float = 2.5,
            patch_size: Optional[int] = None,
            gpu_memory_utilization: float = 0.90,
            max_model_len=8192,
            dtype: str = "auto",
            enforce_eager: bool = False,
            trust_remote_code: bool = True,
            log_path: Optional[str] = None,
            pred_path: Optional[str] = None,
            fail_fast: bool = True,  # batch 出错是否直接 raise
            force_stop_thinking: bool = False,
            **kwargs,
    ):
        self.qwen_version = qwen_version
        self.patch_size = patch_size
        self.fail_fast = fail_fast
        self.force_stop_thinking = force_stop_thinking
        self.pred_path = pred_path

        from vllm import LLM, SamplingParams  # 放这里确保 env 生效
        import qwen_vl_utils
        self.qwen_vl_utils = qwen_vl_utils

        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=trust_remote_code)

        self.sampling_params = SamplingParams(
            temperature=0.0,
            top_p=1.0,
            max_tokens=int(max_new_tokens),
        )

        self.llm = LLM(
            model=model_path,
            trust_remote_code=trust_remote_code,
            max_model_len=max_model_len,
            dtype=dtype,
            gpu_memory_utilization=gpu_memory_utilization,
            enforce_eager=enforce_eager,
            tensor_parallel_size=1,
        )
        os.makedirs(os.path.dirname(self.pred_path), exist_ok=True)

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
            "durations": item.get("durations", None),
            "type": item.get("type", None)
        }
        return payload

    def _append_preds(self, records: List[Dict[str, Any]]):
        with open(self.pred_path, "a", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
            f.flush()

    def _render_prompt(self, prompts: List[Dict[str, Any]]) -> str:
        prompts_str = self.processor.apply_chat_template(
            prompts,
            tokenize=False,
            add_generation_prompt=True,
        )
        if self.force_stop_thinking:
            prompts_str += "</think>"
        return prompts_str

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
                image_inputs_list, video_inputs_list, per_sample_kwargs = process_vision_info_qwen(prompts_batch,
                                                                                                   self.qwen_version,
                                                                                                   self.patch_size)

                vllm_inputs = []
                metas = []

                for i, p in enumerate(batch):
                    prompt_str = self._render_prompt(p["prompt"])
                    one_req = {"prompt": prompt_str}

                    mm_data = {}
                    if image_inputs_list is not None and image_inputs_list[i] is not None:
                        mm_data["image"] = image_inputs_list[i]
                    if video_inputs_list is not None and video_inputs_list[i] is not None:
                        mm_data["video"] = video_inputs_list[i]
                    if mm_data:
                        one_req["multi_modal_data"] = mm_data

                    vk = per_sample_kwargs[i]
                    if vk:
                        one_req["mm_processor_kwargs"] = vk

                    vllm_inputs.append(one_req)
                    metas.append({"idx": p["idx"], "id": p["id"], "target": p["target"], "durations": p["durations"],
                                  "type": p["type"]})

                gen_out = self.llm.generate(vllm_inputs, self.sampling_params)

                batch_records = []
                for m, o in zip(metas, gen_out):
                    text = o.outputs[0].text if o.outputs else ""
                    batch_records.append({"idx": m["idx"], "id": m["id"], "target": m["target"], "pred": text,
                                          "durations": m["durations"], "type": m["type"]})

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
