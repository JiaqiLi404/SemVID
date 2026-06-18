import transformers
import logging
import os
import sys


def setup_logger(name, save_dir, distributed_rank=0, filename="log.json"):
    root = logging.getLogger()  # root logger
    root.setLevel(logging.DEBUG)

    if distributed_rank > 0:
        return root

    formatter = logging.Formatter(
        "%(asctime)s %(name)s %(levelname)s: %(message)s", "%Y-%m-%d %H:%M:%S"
    )

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)
    root.addHandler(ch)

    # supress the massive logs from qwen_vl_utils
    logging.getLogger("qwen_vl_utils.vision_process").setLevel(logging.WARNING)

    # 让 transformers 日志用 python logging，并调到你想要的级别
    transformers.logging.set_verbosity_info()
    # transformers.logging.enable_default_handler()
    transformers.logging.enable_explicit_format()

    # 关键：把 transformers 的 logger 也挂到 root 的 handler
    tf_logger = logging.getLogger("transformers")
    tf_logger.setLevel(logging.INFO)
    tf_logger.propagate = True

    trl_logger = logging.getLogger("trl")
    trl_logger.setLevel(logging.INFO)
    trl_logger.propagate = True

    if save_dir:
        fh = logging.FileHandler(os.path.join(save_dir, filename))
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        root.addHandler(fh)

    return root
