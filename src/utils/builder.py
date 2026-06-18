from __future__ import annotations

from mmengine.registry import Registry,build_from_cfg
from typing import Any, Type, Union

CONFIGS = Registry("configs")

def build_config(cfg):
    return CONFIGS.build(cfg)

def build_cls(name: Union[str, Type[Any]], registry=CONFIGS) -> Type[Any]:
    """
    Build class by name.

    Rules:
      - if `name` is a class: return it
      - if `name` has no '.': return registry.get(name)
      - if `name` contains '.': import and return the class
        supports both "pkg.mod.Class" and "pkg.mod:Class"
    """
    if isinstance(name, type):
        return name

    if not isinstance(name, str) or not name:
        raise TypeError(f"`name` must be a non-empty str or a class, got: {type(name)}")

    # 1) Registry lookup
    if "." not in name and ":" not in name:
        cls = registry.get(name)
        if cls is None:
            available = list(registry.module_dict.keys()) if hasattr(registry, "module_dict") else []
            raise KeyError(f"Class '{name}' not found in registry '{registry.name}'. "
                           f"Available keys (partial): {available[:50]}")
        return cls

    # 2) Import by path
    # allow "a.b:Cls" or "a.b.Cls"
    if ":" in name:
        module_path, cls_name = name.split(":", 1)
    else:
        module_path, cls_name = name.rsplit(".", 1)

    if not module_path or not cls_name:
        raise ValueError(f"Invalid import path: '{name}'. Expected 'pkg.module:Class' or 'pkg.module.Class'")

    try:
        module = __import__(module_path, fromlist=[cls_name])
    except Exception as e:
        raise ImportError(f"Failed to import module '{module_path}' from '{name}': {e}") from e

    try:
        cls = getattr(module, cls_name)
    except AttributeError as e:
        raise ImportError(f"Module '{module_path}' has no attribute '{cls_name}' (from '{name}')") from e

    if not isinstance(cls, type):
        raise TypeError(f"Imported '{name}' but got a non-class object: {cls} (type={type(cls)})")

    return cls