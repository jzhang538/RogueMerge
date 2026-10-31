import os
import random

from evaluation.tasks.generation.base import AdversarialTaskBase
from src.data_utils import load_standard_data


class JailbreakBench(AdversarialTaskBase):
    def __init__(
            self,
            classifier_model_name: str,
            random_seed: int = 42,
            sample_size: int = 100,
            local_model_path: str | None = None,
            data_filename: str = "../../../../../data/safety_eval/llmlat.jsonl"):
        self.random_seed = random_seed
        self.sample_size = sample_size
        self.data_filename = data_filename
        super().__init__(classifier_model_name, local_model_path)

    def load(self) -> list[dict]:
        current_dir = os.path.abspath(os.path.dirname(__file__))
        task_file = os.path.join(current_dir, self.data_filename)
        data = load_standard_data(task_file)

        if self.sample_size <= 0:
            raise ValueError(f"sample_size must be positive, got {self.sample_size}")
        if self.sample_size > len(data):
            raise ValueError(
                f"sample_size ({self.sample_size}) cannot be larger than dataset size ({len(data)})")

        sampled_data = random.Random(self.random_seed).sample(data, self.sample_size)

        for item in sampled_data:
            item["instruction"] = item["original_question"]

        return sampled_data
