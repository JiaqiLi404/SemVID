import re
from typing import List, Optional

import numpy as np
from rouge_score import rouge_scorer

from .base import BaseReward
from ..builder import FUNCTIONS


@FUNCTIONS.register_module()
class ThinkingFormatReward(BaseReward):
    def __init__(self, thinking_pattern=r".*?</think>\s*<answer>.*?</answer>", weight=1.0, **kwargs):
        super(ThinkingFormatReward, self).__init__()
        self.thinking_pattern = re.compile(thinking_pattern, re.DOTALL)
        self.weight = weight

    def __call__(self, completions: List[str], **kwargs) -> List[float]:
        pattern = re.compile(self.thinking_pattern, re.DOTALL)
        matches = [re.fullmatch(pattern, content.strip()) for content in completions]
        return [self.weight if match else 0.0 for match in matches]


@FUNCTIONS.register_module()
class TimestepPairReward(BaseReward):
    def __init__(self,
                 thinking_pattern=r"(.*?)</think>",
                 timestep_pattern=r"<timestep>\s*(\d+\.?\d*)\s+to\s+(\d+\.?\d*)\s*</timestep>",
                 weight=0.2,
                 max_count: int = 1,
                 **kwargs):
        super(TimestepPairReward, self).__init__()
        self.thinking_pattern = re.compile(thinking_pattern, re.DOTALL)
        self.timestep_pattern = re.compile(timestep_pattern, re.IGNORECASE | re.DOTALL,)
        self.weight = weight
        self.max_count = max_count

    def __call__(self, completions: List[str], **kwargs) -> List[float]:
        rewards = []

        for completion in completions:
            score = 0.0
            matches = self.thinking_pattern.findall(completion)
            think_content = matches[-1].strip() if matches else None

            if think_content:
                pair_matches = self.timestep_pattern.findall(think_content)
                pair_count = len(pair_matches)
                capped_count = min(pair_count, self.max_count)
                score = self.weight * capped_count

            rewards.append(max(0.0, score))

        return rewards


@FUNCTIONS.register_module()
class ThinkingLengthReward(BaseReward):
    def __init__(self, thinking_pattern=r"(.*?)</think>", weight=0.5, max_length=180,**kwargs):
        super(ThinkingLengthReward, self).__init__()
        self.thinking_pattern = re.compile(thinking_pattern, re.DOTALL)
        self.reward_per_token = weight / max_length
        self.max_length = max_length

    def __call__(self, completions: List[str], **kwargs) -> List[float]:
        rewards = []
        for completion in completions:
            score = 0.0
            matches = self.thinking_pattern.findall(completion)
            think_content = matches[-1].strip() if matches else None

            if think_content:
                think_length = len(think_content)
                capped_length = min(think_length, self.max_length)
                score = self.reward_per_token * capped_length

            rewards.append(max(0.0, score))

        return rewards


@FUNCTIONS.register_module()
class KeywordUsageReward(BaseReward):
    def __init__(self, thinking_pattern=r"(.*?)</think>", keywords=None, weight: float = 0.1,
                 max_count: int = 2, **kwargs):
        super(KeywordUsageReward, self).__init__()
        self.thinking_pattern = re.compile(thinking_pattern, re.DOTALL)
        self.weight = weight
        self.max_count = max_count

        DEFAULT_STRUCTURE_KEYWORDS = [
            "analyze",
            "compare",
            "deduce",
            "however",
            "therefore",
            "because",
            "step",
            "observe",
            "notice",
            "identify",
            "wait",
        ]
        self.keywords = DEFAULT_STRUCTURE_KEYWORDS if keywords is None else keywords

    def __call__(self, completions: List[str], **kwargs) -> List[float]:
        rewards = []

        for completion in completions:
            score = 0.0
            matches = self.thinking_pattern.findall(completion)
            think_content = matches[-1].strip() if matches else None

            if think_content:
                content_lower = think_content.lower()
                keyword_count = sum(1 for word in self.keywords if word in content_lower)
                capped_count = min(keyword_count, self.max_count)
                score = self.weight * capped_count

            rewards.append(max(0.0, score))

        return rewards


@FUNCTIONS.register_module()
class ParagraphStructureReward(BaseReward):
    def __init__(self, thinking_pattern=r"(.*?)</think>", weight: float = 0.05, max_paragraphs: int = 2,
                 **kwargs):
        super(ParagraphStructureReward, self).__init__()
        self.thinking_pattern = re.compile(thinking_pattern, re.DOTALL)
        self.weight = weight
        self.max_paragraphs = max_paragraphs

    def __call__(self, completions: List[str], **kwargs) -> List[float]:
        rewards = []
        for completion in completions:
            score = 0.0
            matches = self.thinking_pattern.findall(completion)
            think_content = matches[-1].strip() if matches else None

            if think_content:
                paragraphs = [p for p in think_content.split("\n") if p.strip()]
                capped_paragraphs = min(len(paragraphs), self.max_paragraphs)
                score = self.weight * capped_paragraphs

            rewards.append(max(0.0, score))

        return rewards


@FUNCTIONS.register_module()
class DiversityReward(BaseReward):
    def __init__(self, weight: float = 1.0, num_generations: int = 8, **kwargs):
        super(DiversityReward, self).__init__()
        self.weight = weight
        self.num_generations = num_generations

    def __call__(self, completions: List[str], **kwargs) -> List[float]:
        if not completions:
            return []

        batch_size = len(completions) // self.num_generations
        diversity_rewards = []
        scorer = rouge_scorer.RougeScorer(
            ["rougeL"], use_stemmer=True
        )

        for i in range(batch_size):
            group_start_idx = i * self.num_generations
            group_end_idx = (i + 1) * self.num_generations
            current_group_completions = completions[group_start_idx:group_end_idx]

            group_rewards = np.zeros(self.num_generations)
            for j in range(self.num_generations):
                total_dissimilarity = 0
                count = 0
                for k in range(self.num_generations):
                    if j == k:
                        continue
                    try:
                        # rouge_score expects strings, handle potential non-string content if necessary
                        score = scorer.score(
                            str(current_group_completions[j]),
                            str(current_group_completions[k]),
                        )["rougeL"].fmeasure
                        total_dissimilarity += 1.0 - score
                        count += 1
                    except Exception as e:
                        print(
                            f"Warning: Error calculating ROUGE score: {e}. Skipping pair."
                        )
                        # Handle potential errors gracefully, e.g., assign neutral dissimilarity

                if count > 0:
                    group_rewards[j] = total_dissimilarity / count
                else:  # Handle case with only one generation or all others failed
                    group_rewards[j] = 0.0

            diversity_rewards.extend(group_rewards.tolist())

        return diversity_rewards
