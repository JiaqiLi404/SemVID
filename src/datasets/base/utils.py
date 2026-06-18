from typing import Any, Dict, List, Optional, Sequence
import pandas as pd


def _find_key_ci(d: Dict[str, Any], key: str) -> Optional[str]:
    """Find actual key in dict with case-insensitive match."""
    if key in d:
        return key
    lk = key.lower()
    for k in d.keys():
        if k.lower() == lk:
            return k
    return None


def _first_present_ci(d: Dict[str, Any], keys: Sequence[str]) -> Optional[str]:
    """Return the first existing key in `d` (case-insensitive), in priority order."""
    for k in keys:
        real = _find_key_ci(d, k)
        if real is not None and d.get(real) is not None:
            return real
    return None


def _to_list_of_str(x: Any) -> List[str]:
    """Convert ndarray/Series/list/tuple/str to List[str] robustly."""
    # pandas / numpy containers
    try:
        import numpy as np  # optional
        if isinstance(x, np.ndarray):
            return [str(v) for v in x.tolist()]
    except Exception:
        pass

    if isinstance(x, pd.Series):
        return [str(v) for v in x.tolist()]

    # already a sequence
    if isinstance(x, (list, tuple)):
        return [str(v) for v in x]

    # single string or other scalar
    if x is None:
        return []
    return [str(x)]


def _normalize_sentence(x: Any) -> str:
    """Ensure sentence is a string; if list, join."""
    if x is None:
        return ""
    if isinstance(x, str):
        return x
    if isinstance(x, (list, tuple)):
        parts = [str(v) for v in x if v is not None]
        if len(parts) == 1:
            return parts[0]
        return "\n".join(parts)
    return str(x)


def align_record_names(record: Dict[str, Any], drop_source_keys: bool = True) -> Dict[str, Any]:
    """
    Align various alias field names to target names:
      - videoID / video_id / video_name / ... -> video (priority)
      - duration: if value is str => set type=duration_value and (optionally) drop duration
                 else keep duration as-is
      - question / questions / query -> sentence
      - options / option -> choices
      - answer -> solution
    """
    r = dict(record)  # don't mutate input

    # 1) video alignment (priority)
    # You can extend this list if you have more aliases.
    video_src = _first_present_ci(r, ["video", "video_path", "videoID", "video_id", "video_name"])
    if "video" not in r and video_src is not None:
        r["video"] = r.get(video_src)

    if drop_source_keys:
        for k in [ "video_path","videoID", "video_id", "video_name"]:
            real = _find_key_ci(r, k)
            if real is not None and real != "video":
                r.pop(real, None)

    # 2) duration -> type if duration is str, else keep duration
    dur_key = _find_key_ci(r, "duration")
    if dur_key is not None:
        dur_val = r.get(dur_key)
        if isinstance(dur_val, str):
            # only set type if absent, unless you want to override
            if _find_key_ci(r, "type") is None:
                r["type"] = dur_val
            if drop_source_keys:
                r.pop(dur_key, None)

    # 3) question/questions/query -> sentence (priority)
    sent_src = _first_present_ci(r, ["sentence", "question", "questions", "query"])
    if _find_key_ci(r, "sentence") is None and sent_src is not None:
        r["sentence"] = _normalize_sentence(r.get(sent_src))

    if drop_source_keys:
        for k in ["question", "questions", "query"]:
            real = _find_key_ci(r, k)
            if real is not None and real != "sentence":
                r.pop(real, None)

    # 4) options/option -> choices (priority)
    choices_src = _first_present_ci(r, ["choices", "options", "option","candidates"])
    if _find_key_ci(r, "choices") is None and choices_src is not None:
        r["choices"] = _to_list_of_str(r.get(choices_src))

    if drop_source_keys:
        for k in ["options", "option","candidates"]:
            real = _find_key_ci(r, k)
            if real is not None and real != "choices":
                r.pop(real, None)

    # 5) answer -> solution
    sol_src = _first_present_ci(r, ["solution", "answer","correct_choice"])
    if _find_key_ci(r, "solution") is None and sol_src is not None:
        r["solution"] = r.get(sol_src)

    if drop_source_keys:
        for k in ["answer", "correct_choice"]:
            real = _find_key_ci(r, k)
            if real is not None and real != "solution":
                r.pop(real, None)

    # 6) type -> type
    sol_src = _first_present_ci(r, ["type", "duration_group"])
    if _find_key_ci(r, "type") is None and sol_src is not None:
        r["type"] = r.get(sol_src)

    if drop_source_keys:
        for k in ["duration_group"]:
            real = _find_key_ci(r, k)
            if real is not None and real != "type":
                r.pop(real, None)


    return r
