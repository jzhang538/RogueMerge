import os

from evaluation.tasks.generation.base import AdversarialTaskBase
from src.data_utils import load_standard_data


class DoAnythingNow(AdversarialTaskBase):
    def __init__(
            self,
            classifier_model_name: str,
            local_model_path: str | None = None,
            data_filename: str = "../../../../../data/safety_eval/do_anything_now_jailbreak.jsonl"):
        self.data_filename = data_filename
        super().__init__(classifier_model_name, local_model_path)

    def load(self) -> list[dict]:
        current_dir = os.path.abspath(os.path.dirname(__file__))
        task_file = os.path.join(current_dir, self.data_filename)
        data = load_standard_data(task_file)
        for item in data:
            item['instruction'] = item['adversarial']
        return data
