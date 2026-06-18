from mmengine.registry import Registry, build_from_cfg

from src.config import build_merged_config

MODELS = Registry("models")

LOSSES = MODELS
TRAINERS = MODELS
FUNCTIONS = MODELS
EVALUATORS = MODELS
COLLATORS=MODELS

def build_loss(cfg):
    """Build loss."""
    return LOSSES.build(cfg)

def build_trainer(cfg, default_args=None):
    """Build trainer from configs."""
    return build_from_cfg(cfg, TRAINERS, default_args)

def build_function(cfg):
    """Build functions."""
    return FUNCTIONS.build(cfg)

def build_evaluator(cfg):
    """Build evaluator."""
    return EVALUATORS.build(cfg)

def build_collator(cfg):
    """Build collator."""
    return COLLATORS.build(cfg)
