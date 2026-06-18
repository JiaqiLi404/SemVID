import os
import shutil
import logging
import inspect
import importlib
from collections import Counter

import torch
from peft import get_peft_model, PeftModel
from torch import nn
from transformers import AutoConfig, AutoProcessor, AutoTokenizer

try:
    from transformers import AutoModelForImageTextToText as AutoModelForMultimodalGeneration
except ImportError:
    from transformers import AutoModelForVision2Seq as AutoModelForMultimodalGeneration

# monkey patching _validate_model_kwargs to skip raising unused parameters error
from transformers.generation.utils import GenerationMixin
from transformers.modeling_utils import PreTrainedModel
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

GenerationMixin._validate_model_kwargs = lambda self, model_kwargs: None

logger = logging.getLogger(__name__)

AUTO_MODEL_AUTO_MAP_KEYS = (
    "AutoModelForImageTextToText",
    "AutoModelForVision2Seq",
)


def _patch_auto_map_for_handler(cfg, module_ref):
    auto_map = dict(getattr(cfg, "auto_map", {}) or {})
    for auto_model_key in AUTO_MODEL_AUTO_MAP_KEYS:
        auto_map[auto_model_key] = module_ref
    cfg.auto_map = auto_map


def _compute_default_rope_parameters(config=None, device=None, seq_len=None, layer_type=None):
    rope_parameters = getattr(config, "rope_parameters", None)
    if layer_type is not None and isinstance(rope_parameters, dict) and layer_type in rope_parameters:
        rope_parameters = rope_parameters[layer_type]
    elif not isinstance(rope_parameters, dict):
        rope_parameters = getattr(config, "rope_scaling", None) or {}

    base = rope_parameters.get("rope_theta", getattr(config, "rope_theta", 10000.0))
    partial_rotary_factor = rope_parameters.get(
        "partial_rotary_factor", getattr(config, "partial_rotary_factor", 1.0)
    )
    head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
    dim = int(head_dim * partial_rotary_factor)
    inv_freq = 1.0 / (
        base ** (torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / dim)
    )
    return inv_freq, 1.0


ROPE_INIT_FUNCTIONS.setdefault("default", _compute_default_rope_parameters)


def _named_modules(module, **kwargs):
    try:
        return module.named_modules(**kwargs)
    except TypeError:
        return module.named_modules()


def _find_input_embeddings_weight_name(model):
    try:
        input_embeddings = model.get_input_embeddings()
    except (AttributeError, NotImplementedError):
        input_embeddings = None
    if input_embeddings is None:
        return None

    for name, module in _named_modules(model, remove_duplicate=False):
        if module is input_embeddings:
            return f"{name}.weight" if name else "weight"
    return None


def _patch_legacy_tied_weights_keys():
    original = getattr(PreTrainedModel, "get_expanded_tied_weights_keys", None)
    if original is None or getattr(original, "_actionvlm_legacy_tied_weights_compat", False):
        return

    def get_expanded_tied_weights_keys_compat(self, all_submodels: bool = False):
        tied_mapping = getattr(self, "_tied_weights_keys", None)
        if not isinstance(tied_mapping, (list, tuple, set)):
            return original(self, all_submodels=all_submodels)

        if all_submodels:
            expanded_tied_weights = {}
            for prefix, submodule in _named_modules(self, remove_duplicate=False):
                if isinstance(submodule, PreTrainedModel):
                    submodel_tied_weights = submodule.get_expanded_tied_weights_keys(all_submodels=False)
                    if prefix:
                        submodel_tied_weights = {
                            f"{prefix}.{k}": f"{prefix}.{v}" for k, v in submodel_tied_weights.items()
                        }
                    expanded_tied_weights.update(submodel_tied_weights)
            return expanded_tied_weights

        if not getattr(self.config, "tie_word_embeddings", False):
            return {}

        source_weight_name = _find_input_embeddings_weight_name(self)
        if source_weight_name is None:
            return {}

        return {target_name: source_weight_name for target_name in tied_mapping if isinstance(target_name, str)}

    get_expanded_tied_weights_keys_compat._actionvlm_legacy_tied_weights_compat = True
    PreTrainedModel.get_expanded_tied_weights_keys = get_expanded_tied_weights_keys_compat


def _patch_legacy_rotary_embedding_init():
    original = getattr(PreTrainedModel, "_init_weights", None)
    if original is None or getattr(original, "_actionvlm_legacy_rotary_init_compat", False):
        return

    def _init_weights_compat(self, module):
        if (
                "RotaryEmbedding" in module.__class__.__name__
                and hasattr(module, "original_inv_freq")
                and hasattr(module, "inv_freq")
                and getattr(module, "rope_type", None) == "default"
                and not hasattr(module, "compute_default_rope_parameters")
        ):
            inv_freq, _ = _compute_default_rope_parameters(module.config, device=module.inv_freq.device)
            module.inv_freq.copy_(inv_freq)
            if hasattr(module.original_inv_freq, "copy_"):
                module.original_inv_freq.copy_(inv_freq)
            return

        return original(self, module)

    _init_weights_compat._actionvlm_legacy_rotary_init_compat = True
    PreTrainedModel._init_weights = _init_weights_compat


def _infer_generation_cache_position(input_ids, inputs_embeds=None, next_sequence_length=None, past_key_values=None):
    input_tensor = input_ids if input_ids is not None else inputs_embeds
    if input_tensor is None:
        return None

    sequence_length = next_sequence_length if next_sequence_length is not None else input_tensor.shape[1]
    past_seen_tokens = 0
    if past_key_values is not None and hasattr(past_key_values, "get_seq_length"):
        past_seen_tokens = past_key_values.get_seq_length()
        if past_seen_tokens is None:
            past_seen_tokens = 0

    return torch.arange(sequence_length, device=input_tensor.device, dtype=torch.long) + past_seen_tokens


def _patch_legacy_prepare_inputs_for_generation(model):
    prepare_inputs = getattr(model, "prepare_inputs_for_generation", None)
    if prepare_inputs is None or getattr(prepare_inputs, "_actionvlm_cache_position_compat", False):
        return
    if "cache_position" not in inspect.signature(prepare_inputs).parameters:
        return

    def prepare_inputs_for_generation_compat(input_ids, *args, **kwargs):
        if kwargs.get("cache_position") is None:
            cache_position = _infer_generation_cache_position(
                input_ids,
                inputs_embeds=kwargs.get("inputs_embeds"),
                next_sequence_length=kwargs.get("next_sequence_length"),
                past_key_values=kwargs.get("past_key_values"),
            )
            if cache_position is not None:
                kwargs["cache_position"] = cache_position

        return prepare_inputs(input_ids, *args, **kwargs)

    prepare_inputs_for_generation_compat._actionvlm_cache_position_compat = True
    model.prepare_inputs_for_generation = prepare_inputs_for_generation_compat


def _patch_qwen2_rmsnorm_import():
    try:
        from transformers.models.qwen2 import modular_qwen2
    except Exception:
        return

    if hasattr(modular_qwen2, "Qwen2RMSNorm"):
        return

    try:
        from transformers.models.qwen2.modeling_qwen2 import Qwen2RMSNorm
    except Exception:
        return

    modular_qwen2.Qwen2RMSNorm = Qwen2RMSNorm


_patch_legacy_tied_weights_keys()
_patch_legacy_rotary_embedding_init()
_patch_qwen2_rmsnorm_import()


def load_hosted_model(model_path, model_handler=None, model_hyper_parameters=None, dtype="auto", backend="deepspeed",
                      model_init_kwargs=None, peft_config=None, max_memory_ratio=0.95):
    if model_hyper_parameters is None:
        model_hyper_parameters = {}
    if model_init_kwargs is None:
        model_init_kwargs = {}

    if dtype == "auto" or dtype is None:
        torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    elif dtype in ("bf16", "bfloat16"):
        torch_dtype = torch.bfloat16
    elif dtype in ("fp16", "float16"):
        torch_dtype = torch.float16
    elif dtype in ("fp32", "float32"):
        torch_dtype = torch.float32
    elif isinstance(dtype, torch.dtype):
        torch_dtype = dtype
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")

    if backend == "vllm":
        log = print
    else:
        log = logger.info
    log("==================================== Model Loading ====================================")

    loading_args = dict(model_init_kwargs)
    attn_implementation = model_init_kwargs.pop("attn_implementation",
                                                model_hyper_parameters.pop("attn_implementation", "flash_attention_2"))

    model_class = AutoModelForMultimodalGeneration
    if model_handler is not None:
        log(f"Loading model from {model_path} with handler {model_handler}, dtype {torch_dtype}, and attn_implementation {attn_implementation}")
        if "." not in model_handler:
            raise ValueError(
                "model_handler must be like '<file_name>.<handler_name>', "
                "e.g. 'modeling_qwen3_vl_fastvid.Qwen3VLForConditionalGenerationFastVID'"
            )
        file_name, handler_name = model_handler.split(".", 1)
        module_name = f"src.models.models.{file_name}"
        try:
            module = importlib.import_module(module_name)
            model_class = getattr(module, handler_name)
        except (ImportError, AttributeError) as error:
            raise ImportError(f"Cannot load model handler '{module_name}.{handler_name}': {error}") from error

        # Import the local class directly so model_path can be a Hugging Face ID or a local path.
        cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        cfg.architectures = [handler_name]

        loading_args.update({
            'pretrained_model_name_or_path': model_path,
            'trust_remote_code': True,
            'config': cfg,
            'dtype': torch_dtype,
            'device_map': "auto",
        })
    else:
        log(f"Loading model from {model_path} without handler and with dtype {torch_dtype}, attn_implementation {attn_implementation}")
        cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        loading_args.update({
            'pretrained_model_name_or_path': model_path,
            'trust_remote_code': True,
            "config": cfg,
            'dtype': torch_dtype,
            'device_map': "auto",
        })

    for k, v in model_hyper_parameters.items():
        setattr(cfg, k, v)

    setattr(cfg, "attn_implementation", attn_implementation)
    setattr(cfg, "_attn_implementation", attn_implementation)

    if backend == "deepspeed":
        loading_args.pop("device_map")

    max_memory = {}
    for i in range(torch.cuda.device_count()):
        total = torch.cuda.get_device_properties(i).total_memory
        max_memory[i] = int(total * max_memory_ratio)  # bytes
    # 可选：允许 offload 到 CPU（不想 offload 可以删掉）
    max_memory["cpu"] = "32GiB"
    loading_args['max_memory'] = max_memory

    log(f"model_init_kwargs: {loading_args}")
    model = model_class.from_pretrained(**loading_args)
    _patch_legacy_prepare_inputs_for_generation(model)

    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

    if getattr(model.config, "text_config", None) is not None:
        if model.config.text_config.pad_token_id is None:
            model.config.text_config.pad_token_id = processor.tokenizer.pad_token_id

    processor.pad_token_id = processor.tokenizer.pad_token_id
    processor.bos_token_id = processor.tokenizer.bos_token_id
    processor.eos_token_id = processor.tokenizer.eos_token_id

    if peft_config is not None:
        # Analysis the model structure for LoRA config
        names = []
        for n, m in model.named_modules():
            if isinstance(m, nn.Linear):
                names.append(n)
        log(f"The number of linear modules in model: {len(names)}", )
        log(f"The linear modules include: {Counter([x.split('.')[-1] for x in names]).most_common(30)}", )
        log(f"The lora target modules: {', '.join(peft_config.target_modules)}", )
        model = get_peft_model(model, peft_config)
        _patch_legacy_prepare_inputs_for_generation(model)
        model.print_trainable_parameters()

    log("=======================================================================================")
    return model, processor


def merge_model(base_model_path, adapter_path, merge_out_path):
    # assert if path is not a dir
    assert not os.path.exists(merge_out_path) or os.path.isdir(
        merge_out_path), f"Model merge path {merge_out_path} is not a directory."
    os.makedirs(merge_out_path, exist_ok=True)
    if len(os.listdir(merge_out_path)) != 0:
        logger.info(f"[MERGE] merge_out_path {merge_out_path} is not empty, skip merging.")
        return
    assert os.path.exists(adapter_path), f"Adapter path {adapter_path} does not exist."
    assert len(os.listdir(adapter_path)) != 0, f"Adapter path {adapter_path} is empty."

    logger.info(f"[MERGE] base_model_path={base_model_path}")
    logger.info(f"[MERGE] adapter_path={adapter_path}")
    logger.info(f"[MERGE] merge_out_path={merge_out_path}")

    base_model = AutoModelForMultimodalGeneration.from_pretrained(
        base_model_path,
        dtype=torch.bfloat16,
        device_map="cpu",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    processor = AutoProcessor.from_pretrained(base_model_path, trust_remote_code=True)
    processor.save_pretrained(merge_out_path)

    model = PeftModel.from_pretrained(base_model, adapter_path)
    model = model.merge_and_unload()

    model.save_pretrained(merge_out_path, safe_serialization=True)
    tok = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    tok.save_pretrained(merge_out_path)
