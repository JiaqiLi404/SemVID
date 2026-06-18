import glob
import json
import os
import random
import sys
import shutil
import time

import numpy as np
import argparse
from typing import List, Optional, Any, Dict, Type, TypeVar, Tuple

os.environ["USE_TF"] = "0"
os.environ["USE_TORCH"] = "1"
sys.dont_write_bytecode = True
path = os.path.join(os.path.dirname(__file__), "..")
if path not in sys.path:
    sys.path.insert(0, path)
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

from src.config import (
    apply_overrides,
    build_merged_config,
    evaluation_config_errors,
    rebase_data_paths,
    set_runtime_overrides,
)
from src.models.builder import EVALUATORS
from src.models import evaluators as _evaluators  # noqa: F401  (registry side effects)
from src.utils import setup_logger, merge_model
from src.datasets import build_dataset
from src.datasets.builder import build_prompt

import torch
import ray

os.environ['TORCHCODEC_NUM_THREADS'] = "8"


def parse_args():
    main_parser = argparse.ArgumentParser(description="Evaluate SemVID on video understanding benchmarks.")
    main_parser.add_argument("config", metavar="FILE", type=str, help="path to config file")
    main_parser.add_argument("--seed", type=int, default=42, help="random seed")
    main_parser.add_argument("--data-root", help="rebase the config's default data/... paths")
    main_parser.add_argument("--model-path", help="Hugging Face model ID or local checkpoint path")
    main_parser.add_argument("--output-dir", help="directory in which evaluation outputs are written")
    main_parser.add_argument("--gpus", type=int, help="number of GPUs/Ray workers")
    main_parser.add_argument("--retention-ratio", type=float, help="visual-token retention ratio in (0, 1]")
    main_parser.add_argument("--sample-amount", type=int, help="evaluate a deterministic subset of N samples")
    main_parser.add_argument(
        "--analyze-evidence",
        action="store_true",
        help="compute ER/CS metrics and save per-sample attention analysis (Qwen3 SemVID)",
    )
    main_parser.add_argument(
        "--analysis-root",
        default="analysis_outputs",
        help="shared directory for full-token baselines and pruned ER/CS outputs",
    )
    main_parser.add_argument(
        "--cfg-options",
        nargs="*",
        default=[],
        metavar="KEY=VALUE",
        help="additional dotted config overrides, e.g. eval_args.eval_batch_size=1",
    )
    main_args = main_parser.parse_args()

    cfg = build_merged_config(main_args.config)
    cfg = rebase_data_paths(cfg, main_args.data_root)
    cfg = set_runtime_overrides(
        cfg,
        model_path=main_args.model_path,
        output_dir=main_args.output_dir,
        eval_gpus=main_args.gpus,
        retention_ratio=main_args.retention_ratio,
        sample_amount=main_args.sample_amount,
    )
    cfg = apply_overrides(cfg, main_args.cfg_options)
    if main_args.analyze_evidence:
        cfg.setdefault("eval_args", {})["analyze_evidence"] = True
        cfg["eval_args"]["analysis_root"] = main_args.analysis_root
    config_errors = evaluation_config_errors(cfg)
    if config_errors:
        formatted = "\n  - ".join(config_errors)
        main_parser.error(f"invalid evaluation config:\n  - {formatted}")

    script_cfg = cfg.get("script_args", {})
    train_cfg = cfg.get("training_args", {})
    model_cfg = cfg.get("model_args", {})
    eval_cfg = cfg.get("eval_args", {})
    return main_args, cfg, script_cfg, train_cfg, model_cfg, eval_cfg


def compute_IoU(pred, gt):
    """Compute the IoU given predicted and ground truth windows."""
    assert isinstance(pred, list) and isinstance(gt, list)
    pred_is_list = isinstance(pred[0], list)
    gt_is_list = isinstance(gt[0], list)
    if not pred_is_list:
        pred = [pred]
    if not gt_is_list:
        gt = [gt]
    pred, gt = np.array(pred), np.array(gt)
    inter_left = np.maximum(pred[:, 0, None], gt[None, :, 0])
    inter_right = np.minimum(pred[:, 1, None], gt[None, :, 1])
    inter = np.maximum(0.0, inter_right - inter_left)
    union_left = np.minimum(pred[:, 0, None], gt[None, :, 0])
    union_right = np.maximum(pred[:, 1, None], gt[None, :, 1])
    union = np.maximum(0.0, union_right - union_left)
    overlap = 1.0 * inter / union
    if not gt_is_list:
        overlap = overlap[:, 0]
    if not pred_is_list:
        overlap = overlap[0]
    return overlap


def _chunk_indices(n: int, num_shards: int) -> List[List[int]]:
    """把 [0..n) 切成 num_shards 份（尽量均匀）。"""
    if num_shards <= 1:
        return [list(range(n))]
    shards = np.array_split(np.arange(n), num_shards)
    return [s.tolist() for s in shards]


def run_eval_debug(eval_dataset, eval_cfg, training_args, logger, work_dir, model_handler,
                   model_hyper_parameters):
    eval_gpus = eval_cfg.get("eval_gpus")
    eval_bs = eval_cfg.get("eval_batch_size", 1)
    max_new_tokens = int(eval_cfg.get("eval_max_new_tokens", 256))
    gpu_memory_utilization = eval_cfg.get("gpu_memory_utilization", 0.8)
    qwen_version = eval_cfg.get("qwen_version", 3)
    max_model_len = eval_cfg.get("max_model_len", 8192)
    force_stop_thinking = eval_cfg.get("force_stop_thinking", False)
    analyze_evidence = bool(eval_cfg.get("analyze_evidence", False))
    analysis_root = eval_cfg.get("analysis_root", "analysis_outputs")
    model_path = eval_cfg.get("eval_model_path")
    if not model_path:
        raise ValueError(
            "model_path is empty. Please set eval_model_merge_path/eval_model_adapter_path or training_args.output_dir")
    eval_gpus = torch.cuda.device_count() if eval_gpus is None else int(eval_gpus)
    logger.info(f"[EVAL] model_path = {model_path}")
    logger.info(f"[EVAL] eval_gpus={eval_gpus}, eval_batch_size={eval_bs}, max_new_tokens={max_new_tokens}")
    eval_gpus = 1
    logger.info(f"[EVAL] eval_gpus={eval_gpus}, eval_batch_size={eval_bs}, max_new_tokens={max_new_tokens}")
    VLLMWorker = EVALUATORS.get(eval_cfg.get("evaluator_type", "QwenVLLMEvaluator"))
    worker = VLLMWorker(
        model_path=model_path,
        model_handler=model_handler,
        model_hyper_parameters=model_hyper_parameters,
        max_new_tokens=max_new_tokens,
        qwen_version=qwen_version,
        trust_remote_code=True,
        dtype="bfloat16",
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        enforce_eager=False,
        log_path=os.path.join(work_dir, f"worker_{0}.log"),
        pred_path=os.path.join(work_dir, f"pred_worker_{0}.jsonl"),
        fail_fast=True,
        force_stop_thinking=force_stop_thinking,
        do_visualize_attn_map=analyze_evidence,
        analysis_root=analysis_root,
    )
    reqs = []
    for idx in range(len(eval_dataset)):
        item = eval_dataset[idx]
        reqs.append(worker.build_driver_payload(item, idx))
    worker.generate_requests(reqs, batch_size=1)


def run_eval_with_ray(eval_dataset, eval_cfg, training_args, logger, work_dir, model_handler,
                      model_hyper_parameters):
    eval_gpus = eval_cfg.get("eval_gpus")
    eval_bs = eval_cfg.get("eval_batch_size", 1)
    max_new_tokens = int(eval_cfg.get("eval_max_new_tokens", 256))
    gpu_memory_utilization = eval_cfg.get("gpu_memory_utilization", 0.8)
    qwen_version = eval_cfg.get("qwen_version", 3)
    max_model_len = eval_cfg.get("max_model_len", 8192)
    force_stop_thinking = eval_cfg.get("force_stop_thinking", False)
    analyze_evidence = bool(eval_cfg.get("analyze_evidence", False))
    analysis_root = eval_cfg.get("analysis_root", "analysis_outputs")
    model_path = eval_cfg.get("eval_model_path")
    if not model_path:
        raise ValueError(
            "model_path is empty. Please set eval_model_merge_path/eval_model_adapter_path or training_args.output_dir")

    visible_gpus = torch.cuda.device_count()
    eval_gpus = visible_gpus if eval_gpus is None else int(eval_gpus)
    if visible_gpus < 1:
        raise RuntimeError("Evaluation requires at least one visible CUDA GPU.")
    if eval_gpus > visible_gpus:
        raise ValueError(
            f"Requested {eval_gpus} GPU workers, but only {visible_gpus} CUDA devices are visible. "
            "Adjust CUDA_VISIBLE_DEVICES or pass a smaller --gpus value."
        )

    logger.info(f"[EVAL] model_path = {model_path}")
    logger.info(f"[EVAL] eval_gpus={eval_gpus}, eval_batch_size={eval_bs}, max_new_tokens={max_new_tokens}")

    # Ray init
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True, log_to_driver=True)

    n = len(eval_dataset)
    logger.info(f"[EVAL] dataset size = {n}")
    shards = _chunk_indices(n, eval_gpus)
    evaluator_class = EVALUATORS.get(eval_cfg.get("evaluator_type", "QwenVLLMEvaluator"))
    if evaluator_class is None:
        raise KeyError(f"Unknown evaluator: {eval_cfg.get('evaluator_type')}")
    remote_evaluator = ray.remote(num_gpus=1)(evaluator_class)

    # 启动 workers
    workers = [
        remote_evaluator.remote(
            model_path=model_path,
            model_handler=model_handler,
            model_hyper_parameters=model_hyper_parameters,
            max_new_tokens=max_new_tokens,
            qwen_version=qwen_version,
            trust_remote_code=True,
            dtype="bfloat16",
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            enforce_eager=False,
            log_path=os.path.join(work_dir, f"worker_{i}.log"),
            pred_path=os.path.join(work_dir, f"pred_worker_{i}.jsonl"),
            fail_fast=True,
            force_stop_thinking=force_stop_thinking,
            do_visualize_attn_map=analyze_evidence,
            analysis_root=analysis_root,
        )
        for i in range(eval_gpus)
    ]

    # 为每个 shard 构造请求（只把必要字段发给 worker，避免传 dataset 对象）
    shard_requests: List[List[Dict[str, Any]]] = []
    for shard_id, idx_list in enumerate(shards):
        reqs = []
        for idx in idx_list:
            item = eval_dataset[idx]
            reqs.append(evaluator_class.build_driver_payload(item, idx))
        shard_requests.append(reqs)
        logger.info(f"[EVAL] shard {shard_id}: {len(reqs)} samples")

    # 并行执行
    futures = []
    t0 = time.time()
    for shard_id, (w, reqs) in enumerate(zip(workers, shard_requests)):
        bs_i = eval_bs[shard_id] if isinstance(eval_bs, (list, tuple)) else eval_bs
        futures.append(w.generate_requests.remote(reqs, batch_size=bs_i))

    worker_summaries: List[Dict[str, Any]] = ray.get(futures)
    num_ok = sum(int(summary.get("num_ok", 0)) for summary in worker_summaries)
    num_err = sum(int(summary.get("num_err", 0)) for summary in worker_summaries)
    logger.info(
        f"[EVAL] generation done, ok={num_ok}, errors={num_err}, "
        f"elapsed={time.time() - t0:.2f}s"
    )

    # Workers append predictions to per-worker JSONL files. Consolidate those
    # records rather than treating each worker's summary dictionary as output.
    results: List[Dict[str, Any]] = []
    worker_paths = sorted(glob.glob(os.path.join(work_dir, "pred_worker_*.jsonl")))
    for worker_path in worker_paths:
        with open(worker_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    results.append(json.loads(line))

    # 保存预测
    pred_path = os.path.join(work_dir, "predictions.jsonl")
    with open(pred_path, "w", encoding="utf-8") as f:
        for record in results:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    logger.info(f"[EVAL] consolidated {len(results)} predictions to: {pred_path}")
    return results


def is_number(s):
    try:
        return float(s)
    except ValueError:
        return None


def main():
    main_args, cfg, script_cfg, training_args, model_args, eval_cfg = parse_args()
    dataset_name = eval_cfg.get("dataset_name")

    # create work_dir and save config
    work_dir = os.path.join(training_args.get("output_dir"), f"eval_{dataset_name}/")
    dir_name = os.path.expanduser(work_dir)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name, mode=0o777, exist_ok=True)
    shutil.copy2(main_args.config, work_dir)
    out_path = os.path.join(work_dir, "config.json")
    with open(out_path, "w") as f:
        json.dump(cfg, f, indent=4)

    # setup logger
    logger = setup_logger(__name__, save_dir=work_dir)
    logger.info(f"Using torch version: {torch.__version__}, CUDA version: {torch.version.cuda}")
    logger.info(f"Config: \n{json.dumps(cfg, ensure_ascii=False, indent=2)}")

    # set random seed
    seed = main_args.seed
    print(f"Using random seed: {seed}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # build dataset
    eval_cfg['is_shuffle'] = False if eval_cfg.get('is_shuffle') is None else eval_cfg.get('is_shuffle')
    eval_dataset = build_dataset(eval_cfg,
                                 default_args=dict(
                                     train_data_path=None,
                                     train_video_folder=None,
                                     is_train=False,
                                     skip_loading_video=True,
                                     logger=logger
                                 ))

    # get the already evaluated samples and filter them
    existing_preds = set()
    for f in os.listdir(work_dir):
        if f.startswith("pred_worker_"):
            with open(os.path.join(work_dir, f), "r", encoding="utf-8") as fin:
                for line in fin:
                    existing_preds.add(json.loads(line)["id"])
    eval_dataset.dataset = [item for item in eval_dataset.dataset if
                            item.get("id", item.get("sample_id")) not in existing_preds]
    logger.info(
        f"Filtered {len(existing_preds)} already evaluated samples. Remaining eval samples: {len(eval_dataset)}")

    # merge model first
    if eval_cfg.get("eval_model_merge_path"):
        base_model_path = eval_cfg.get("eval_model_path")
        adapter_path = eval_cfg.get("eval_model_adapter_path", None)
        merge_out_path = eval_cfg.get("eval_model_merge_path")
        if adapter_path is None:
            adapter_path = training_args.get("output_dir")
            # find the latest checkpoint
            ckpt_list = glob.glob(os.path.join(adapter_path, "checkpoint-*"))
            ckpt_list.sort(key=os.path.getctime)
            if len(ckpt_list) == 0:
                raise ValueError(f"No checkpoint found in {adapter_path}")
            adapter_path = ckpt_list[-1]

        merge_model(base_model_path, adapter_path, merge_out_path)
        eval_cfg['eval_model_path'] = merge_out_path

    if len(eval_dataset) != 0:
        # run_eval_debug(
        run_eval_with_ray(
            eval_dataset=eval_dataset,
            eval_cfg=eval_cfg,
            training_args=training_args,
            logger=logger,
            work_dir=work_dir,
            model_handler=script_cfg.get("model_handler", None),
            model_hyper_parameters=script_cfg.get("model_hyper_parameters", None),
        )
        logger.info("Prediction is done. Starting evaluation.")
    else:
        logger.info("No more samples to evaluate, skipping evaluation.")

    # evaluate the IoU
    data = []
    logs = []
    for f in os.listdir(work_dir):
        if f.startswith("pred_worker_"):
            with open(os.path.join(work_dir, f), "r", encoding="utf-8") as fin:
                for line in fin:
                    data.append(json.loads(line))
        elif f.startswith("worker_") and f.endswith(".log"):
            with open(os.path.join(work_dir, f), "r", encoding="utf-8") as fin:
                for line in fin:
                    if '[Metrics]' not in line:
                        continue
                    line = line.strip().split('[Metrics]')[1]
                    logs.append(json.loads(line))

    if not data:
        raise RuntimeError(
            "No predictions are available for scoring. Check that the annotation file contains "
            "samples and that the configured video paths exist."
        )

    task = eval_cfg.get("prompt_task")
    prompt_cls = build_prompt(eval_cfg.get("prompt_cls"))
    decode_func = getattr(prompt_cls, f"decode_for_{task}")

    if task == "temporal_grounding":
        for d in data:
            id = d["id"]
            if "target" in d:
                start_time = d["target"][0]
                end_time = d["target"][1]
            else:
                start_time = float(id.split("_")[-2])
                end_time = float(id.split("_")[-1])
            pred_str = d['pred']
            res = decode_func(pred_str)
            d['res'] = res
            pred_start_time, pred_end_time = res[0], res[1]
            score = 0.0
            if pred_start_time is not None and pred_end_time is not None:
                score = compute_IoU([pred_start_time, pred_end_time], [start_time, end_time])
            d["score"] = score

        results = {}
        logger.info(f"dataset {eval_cfg.get('dataset_name')} has {len(data)} samples")
        # calculate scores
        results['mIoU'] = sum([d['score'] for d in data]) / len(data) * 100
        logger.info(f"mIoU: {results['mIoU']}")
        avg_iou = []
        for threshold in [0.3, 0.5, 0.7]:
            iou = sum([1 for d in data if d['score'] >= threshold]) / len(data) * 100
            results[f"IoU R1@{threshold}"] = iou
            logger.info(f"IoU R1@{threshold}: {iou}")
            avg_iou.append(iou)
        results['avg'] = sum(avg_iou) / len(avg_iou)
        logger.info(f"Avg IoU: {results['avg']}")
        # calculate metrics
        if len(logs) > 0:
            metrics = {}
            for log in logs:
                for k, v in log.items():
                    v = is_number(v)
                    if v is None:
                        continue
                    if k not in metrics:
                        metrics[k] = []
                    metrics[k].append(v)
            results['metrics'] = {'mean': {}, 'std': {}, 'max': {}, 'min': {}}
            for k, v in metrics.items():
                results['metrics']['mean'][k] = sum(v) / len(v)
                results['metrics']['std'][k] = np.std(np.array(v))
                results['metrics']['max'][k] = max(v)
                results['metrics']['min'][k] = min(v)
        json.dump(results, open(os.path.join(work_dir, "scores.json"), "w"), indent=4)

        data.sort(key=lambda x: x["score"], reverse=True)
        with open(os.path.join(work_dir, "predictions_with_score.json"), "w", encoding="utf-8") as f:
            for d in data:
                f.write(json.dumps(d, ensure_ascii=False) + "\n")
    elif task == "multichoice_qa":
        types = set()
        for d in data:
            id = d["id"]
            answer_gt = d['target']
            pred_str = d['pred']
            types.add(d['type']) if 'type' in d and d['type'] is not None else None
            res = decode_func(pred_str)
            if isinstance(res, str):
                res = [res]
            if isinstance(answer_gt, str):
                answer_gt = [answer_gt]
            if len(res) != len(answer_gt):
                d['score'] = 0
                continue
            d['score'] = 1
            for ans in res:
                if ans not in answer_gt:
                    d['score'] = 0
                    break

        results = {}
        logger.info(f"dataset {eval_cfg.get('dataset_name')} has {len(data)} samples")
        # calculate scores
        results['Overall'] = sum([d['score'] for d in data]) / len(data) * 100
        logger.info(f"Overall: {results['Overall']}")

        if len(types) > 0:
            for ty in types:
                res_ty = [d['score'] for d in data if d['type'] == ty]
                acc_ty = sum(res_ty) / len(res_ty) if len(res_ty) > 0 else 0
                results[f"ACC {ty}"] = acc_ty
                logger.info(f"ACC {ty}: {acc_ty}")
        # calculate metrics
        if len(logs) > 0:
            metrics = {}
            for log in logs:
                for k, v in log.items():
                    v = is_number(v)
                    if v is None:
                        continue
                    if k not in metrics:
                        metrics[k] = []
                    metrics[k].append(v)
            results['metrics'] = {'mean': {}, 'std': {}, 'max': {}, 'min': {}}
            for k, v in metrics.items():
                results['metrics']['mean'][k] = sum(v) / len(v)
                results['metrics']['std'][k] = np.std(np.array(v))
                results['metrics']['max'][k] = max(v)
                results['metrics']['min'][k] = min(v)
        json.dump(results, open(os.path.join(work_dir, "scores.json"), "w"), indent=4)

        data.sort(key=lambda x: x["score"], reverse=True)
        with open(os.path.join(work_dir, "predictions_with_score.json"), "w", encoding="utf-8") as f:
            for d in data:
                f.write(json.dumps(d, ensure_ascii=False) + "\n")
    else:
        raise ValueError(f"Unknown task: {task}")


if __name__ == "__main__":
    main()
