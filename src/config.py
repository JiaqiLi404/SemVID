"""Lightweight configuration loading and command-line overrides for SemVID."""

from __future__ import annotations

import ast
import os
import runpy
import types
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Iterable, List, Set, Union


BASIC_VALUE_TYPES = (str, int, float, bool, type(None), dict, list, tuple)


def _is_exportable_value(value: Any) -> bool:
    if isinstance(value, types.ModuleType):
        return False
    if isinstance(value, (types.FunctionType, types.BuiltinFunctionType, types.MethodType, type)):
        return False
    return isinstance(value, BASIC_VALUE_TYPES) or isinstance(value, Mapping)


def _load_python_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    namespace = runpy.run_path(str(config_path))
    return {
        key: value
        for key, value in namespace.items()
        if not key.startswith("__") and _is_exportable_value(value)
    }


def _normalize_base_list(base_value: Any) -> List[str]:
    if base_value is None:
        return []
    if isinstance(base_value, (str, Path)):
        return [str(base_value)]
    if isinstance(base_value, (list, tuple)):
        return [str(item) for item in base_value]
    raise TypeError(f"Unsupported _base_ type: {type(base_value)}")


def deep_merge(base: Any, child: Any) -> Any:
    """Recursively merge dictionaries, with child values taking precedence."""
    if isinstance(child, Mapping) and child.get("_delete_", False) is True:
        replacement = dict(child)
        replacement.pop("_delete_", None)
        return deepcopy(replacement)

    if isinstance(base, Mapping) and isinstance(child, Mapping):
        merged = deepcopy(dict(base))
        for key, child_value in child.items():
            if key == "_delete_":
                continue
            merged[key] = (
                deep_merge(merged[key], child_value)
                if key in merged
                else deepcopy(child_value)
            )
        return merged

    return deepcopy(child)


def build_merged_config(
    config_path: Union[str, Path], visited: Set[str] | None = None
) -> Dict[str, Any]:
    """Load a Python config and recursively merge its ``_base_`` configs."""
    config_path = Path(config_path).expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    visited = visited or set()
    key = str(config_path)
    if key in visited:
        raise RuntimeError(f"Circular _base_ reference detected: {config_path}")
    visited.add(key)

    current = _load_python_config(config_path)
    base_files = _normalize_base_list(current.pop("_base_", None))

    merged: Dict[str, Any] = {}
    for base_file in base_files:
        base_path = Path(base_file).expanduser()
        if not base_path.is_absolute():
            base_path = config_path.parent / base_path
        merged = deep_merge(merged, build_merged_config(base_path, visited=visited))

    visited.remove(key)
    return deep_merge(merged, current)


def parse_override_value(raw_value: str) -> Any:
    """Parse an override value while keeping ordinary unquoted strings convenient."""
    aliases = {"true": True, "false": False, "none": None, "null": None}
    lowered = raw_value.lower()
    if lowered in aliases:
        return aliases[lowered]
    try:
        return ast.literal_eval(raw_value)
    except (SyntaxError, ValueError):
        return raw_value


def apply_overrides(config: Dict[str, Any], options: Iterable[str] | None) -> Dict[str, Any]:
    """Apply ``section.nested_key=value`` overrides to a config dictionary."""
    config = deepcopy(config)
    for option in options or []:
        if "=" not in option:
            raise ValueError(f"Invalid override '{option}'. Expected KEY=VALUE.")
        dotted_key, raw_value = option.split("=", 1)
        keys = [key for key in dotted_key.split(".") if key]
        if not keys:
            raise ValueError(f"Invalid override key in '{option}'.")

        cursor = config
        for key in keys[:-1]:
            value = cursor.setdefault(key, {})
            if not isinstance(value, dict):
                raise TypeError(f"Cannot set '{dotted_key}': '{key}' is not a mapping.")
            cursor = value
        cursor[keys[-1]] = parse_override_value(raw_value)
    return config


def rebase_data_paths(config: Dict[str, Any], data_root: str | None) -> Dict[str, Any]:
    """Rebase default ``data/...`` evaluation paths under a user-provided root."""
    if not data_root:
        return config

    config = deepcopy(config)
    root = Path(data_root).expanduser()
    eval_config = config.setdefault("eval_args", {})
    for key in ("eval_data_path", "eval_video_folder", "subtitle_folder"):
        value = eval_config.get(key)
        if not value:
            continue
        path = Path(value)
        if not path.parts or path.parts[0].lower() != "data":
            continue
        eval_config[key] = str(root.joinpath(*path.parts[1:]))
    return config


def evaluation_config_errors(config: Dict[str, Any], *, check_paths: bool = True) -> List[str]:
    """Return actionable errors for an evaluation config without importing ML packages."""
    errors: List[str] = []
    eval_config = config.get("eval_args", {})
    script_config = config.get("script_args", {})
    training_config = config.get("training_args", {})

    required = {
        "eval_args.type": eval_config.get("type"),
        "eval_args.dataset_name": eval_config.get("dataset_name"),
        "eval_args.eval_data_path": eval_config.get("eval_data_path"),
        "eval_args.eval_video_folder": eval_config.get("eval_video_folder"),
        "eval_args.eval_model_path": eval_config.get("eval_model_path"),
        "script_args.model_handler": script_config.get("model_handler"),
        "training_args.output_dir": training_config.get("output_dir"),
    }
    errors.extend(f"Missing required setting: {key}" for key, value in required.items() if not value)

    if check_paths:
        for key in ("eval_data_path", "eval_video_folder", "subtitle_folder"):
            value = eval_config.get(key)
            if value and not Path(value).expanduser().exists():
                errors.append(f"Path does not exist: eval_args.{key}={value}")

    ratio = script_config.get("model_hyper_parameters", {}).get("semantic_retention_ratio")
    if ratio is not None and not 0 < float(ratio) <= 1:
        errors.append("script_args.model_hyper_parameters.semantic_retention_ratio must be in (0, 1].")
    eval_gpus = eval_config.get("eval_gpus")
    if eval_gpus is not None and int(eval_gpus) < 1:
        errors.append("eval_args.eval_gpus must be at least 1.")
    return errors


def set_runtime_overrides(
    config: Dict[str, Any],
    *,
    model_path: str | None = None,
    output_dir: str | None = None,
    eval_gpus: int | None = None,
    retention_ratio: float | None = None,
    sample_amount: int | None = None,
) -> Dict[str, Any]:
    """Apply the common evaluation overrides exposed by ``tools/evaluate.py``."""
    config = deepcopy(config)
    eval_config = config.setdefault("eval_args", {})
    training_config = config.setdefault("training_args", {})
    hyperparameters = config.setdefault("script_args", {}).setdefault("model_hyper_parameters", {})

    if model_path is not None:
        eval_config["eval_model_path"] = model_path
        config.setdefault("model_args", {})["model_name_or_path"] = model_path
    if output_dir is not None:
        training_config["output_dir"] = output_dir
    if eval_gpus is not None:
        eval_config["eval_gpus"] = eval_gpus
    if retention_ratio is not None:
        hyperparameters["semantic_retention_ratio"] = retention_ratio
    if sample_amount is not None:
        eval_config["sample_amount"] = sample_amount
    return config
