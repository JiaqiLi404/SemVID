import logging
import os
import sys

def setup_root_logging(log_path: str, level=logging.INFO):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    os.environ.setdefault("PYTHONUNBUFFERED", "1")

    fmt = "[%(asctime)s][%(levelname)s][%(name)s]"
    fmt += " %(message)s"

    handlers = [
        logging.FileHandler(log_path, mode="a", encoding="utf-8"),
    ]
    handlers.append(logging.StreamHandler(sys.stdout))

    logging.basicConfig(
        level=level,
        format=fmt,
        handlers=handlers,
        force=True,
    )

    # suppress massive logs
    logging.getLogger("qwen_vl_utils.vision_process").setLevel(logging.WARNING)
