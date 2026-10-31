import pandas as pd
import json
from evaluation.tasks.classification.wildguardtest_prompt import WildGuardTestPromptHarmBase
from src.classifier_models.base import PromptHarmfulness

class WildGuardInjectionEval(WildGuardTestPromptHarmBase):
    def __init__(self, dataset_path: str, **kwargs):
        """
        初始化任务
        :param dataset_path: 本地数据集的绝对路径或相对路径
        """
        self.dataset_path = dataset_path
        super().__init__(**kwargs)

    def load(self) -> list[dict]:
        """
        加载本地数据集并转换为评估框架需要的格式
        """
        data_list = []
        
        # 假设你的本地文件是 JSONL 格式（每行一个 JSON）
        # 如果是 CSV，可以使用 pd.read_csv(self.dataset_path)
        try:
            # 这里以读取 JSONL 为例
            with open(self.dataset_path, 'r', encoding='utf-8') as f:
                for line in f:
                    item = json.loads(line)
                    
                    # -------------------------------------------------
                    # 关键逻辑：数据映射
                    # 你的数据集应该已经包含了修改后的 prompt (ifeval + ignore + wildguard)
                    # 如果没有，你也可以在这里动态拼接
                    # -------------------------------------------------
                    
                    # 1. 提取标签 (假设本地数据保留了原始 wildguard 的 label 字段名)
                    # 需要将字符串标签转换为 PromptHarmfulness 枚举
                    gt_label = item.get("prompt_harm_label", "unharmful") # 根据你的实际字段名修改
                    harmfulness_enum = PromptHarmfulness.HARMFUL if gt_label == "harmful" else PromptHarmfulness.UNHARMFUL
                    
                    # 2. 提取 Prompt (已经是注入过的文本)
                    prompt_text = item.get("prompt") 
                    
                    # 3. 构建返回对象
                    data_list.append({
                        "id": item.get("id", len(data_list)), # 如果没有ID则自动生成
                        "prompt": prompt_text,
                        "response": "", # 分类任务通常不需要 response，除非模型依赖它
                        "prompt_harmfulness": harmfulness_enum,
                        "prompt_type": item.get("prompt_type", "injection"), # 标记为 injection
                        "prompt_harm_category": item.get("subcategory", None)
                    })
                    
        except FileNotFoundError:
            raise FileNotFoundError(f"未找到本地数据集文件: {self.dataset_path}")
            
        return data_list