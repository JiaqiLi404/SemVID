import re
from typing import List, Optional, Tuple, Union, Dict, Any

from ..builder import PROMPTS


@PROMPTS.register_module()
class Qwen3Prompts:
    def __init__(self, thinking=False, **kwargs):
        self.thinking = thinking

    def for_temporal_grounding(self, event, **kwargs):
        thinking = kwargs.get("thinking", self.thinking)
        if thinking:
            res = f"""Give you a textual query: {event}.
When does the described content occur in the video?
Please brief your thought process in one or two short sentences and return the final timestamp in seconds.
"""
        else:
            res = f"""Give you a textual query: {event}.
When does the described content occur in the video?
Please return the final timestamps in seconds directly in one sentence.
"""
        return res

    def for_multichoice_qa(self, event, choices, **kwargs):
        thinking = kwargs.get("thinking", self.thinking)
        if isinstance(choices, list) or isinstance(choices,tuple):
            choices="\n".join(choices)
        if thinking:
            return ("Select the best answer to the following multiple-choice question based on the video. "
                    "Respond with only the letter of the correct option.\n"
                    f"Question: {event}\n"
                    f"{choices}\n"
                    f"Please reason step-by-step, identify relevant visual content, analyze key timestamps and clues, "
                    f"and then provide the final answer.")
        else:
            return ("Select the best answer to the following multiple-choice question based on the video. "
                    "Respond with only the letter of the correct option.\n"
                    f"Question: {event}\n  Possible answer choices:\n"
                    f"{choices}\n"
                    f"The best answer is:")

    # getattr(self.prompt_cls, f"decode_for_{prompt_task}")
    def decode_for_temporal_grounding(self, output_string: str) -> List[Optional[float]]:
        def collect_spans(text: str) -> List[Tuple[float, float]]:
            # Matching：
            # 145.9 seconds to 184.5 seconds
            # 145.9 sec to 184.5
            # 145.9s-184.5s
            # 145.9 to 184.5
            pat = (
                r"(\d+\.?\d*)\s*(?:seconds?|secs?|sec|s)?\s*"
                r"(?:to|and|-)\s*"
                r"(\d+\.?\d*)\s*(?:seconds?|secs?|sec|s)?"
            )

            spans: List[Tuple[float, float]] = []
            for a, b in re.findall(pat, text, flags=re.IGNORECASE):
                try:
                    spans.append((float(a), float(b)))
                except ValueError:
                    pass
            return spans

        texts = [output_string]
        m = re.search(r"<answer>(.*?)</answer>", output_string, flags=re.DOTALL | re.IGNORECASE)
        if m:
            texts.append(m.group(1))

        spans: List[Tuple[float, float]] = []
        for t in texts:
            spans.extend(collect_spans(t))

        if not spans:
            return [None, None]

        s, e = spans[-1]
        return [s, e]

    def decode_for_multichoice_qa(
            self,
            output_string: str,
            choices: Optional[Union[str, List[str], Dict[str, str]]] = None,
            **kwargs: Any,
    ) -> Optional[str]:
        """
        Return 'A'/'B'/'C'/'D' or None.
        - supports outputs like: "\\n\\nA", "Answer: C", "Option (B)", "<answer>D</answer>"
        - optional fallback: match option text if model repeats the content instead of the letter
        """
        # 1) prefer <answer>...</answer>
        texts = [output_string]
        m = re.search(r"<answer>\s*(.*?)\s*</answer>", output_string, flags=re.IGNORECASE | re.DOTALL)
        if m and m.group(1).strip():
            texts.insert(0, m.group(1).strip())

        # 2) extract letter
        LETTER = r"([A-Za-z])"
        patterns = [
            rf"^\s*{LETTER}\s*$",
            rf"\b(?:answer|ans|final|choice|option)\s*[:：\-]?\s*\(?\s*{LETTER}\s*\)?\b",
            rf"\b(?:the\s+best\s+answer\s+is)\s*[:：\-]?\s*\(?\s*{LETTER}\s*\)?\b",
            rf"\b(?:答案|选项|选择)\s*[:：\-]?\s*\(?\s*{LETTER}\s*\)?\b",
            rf"\(\s*{LETTER}\s*\)",
            rf"^\s*{LETTER}[\.\)]\s*",  # "C." or "C)"
            rf"\b{LETTER}\b",
        ]
        for t in texts:
            for p in patterns:
                mm = re.search(p, t, flags=re.IGNORECASE)
                if mm:
                    return mm.group(1).upper()

        return None
