from typing import List, Optional, Tuple

from ..builder import PROMPTS
import re


@PROMPTS.register_module()
class Qwen2Prompts:
    def __init__(self, thinking=False, **kwargs):
        self.thinking = thinking

    def for_temporal_grounding(self, event, **kwargs):
        if self.thinking:
            return f"""To accurately pinpoint the event "{event}" in the video, determine the precise time period of the event.

Output your thought process within the <think> </think> tags, including analysis with either specific time ranges (xx.xx to xx.xx) in <timestep> </timestep> tags.

Then, provide the start and end times (in seconds, precise to two decimal places) in the format "start time to end time" within the <answer> </answer> tags. For example: "12.54 to 17.83"."""

        return f"""To accurately pinpoint the event "{event}" in the video, determine the precise time period of the event in seconds."""

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
