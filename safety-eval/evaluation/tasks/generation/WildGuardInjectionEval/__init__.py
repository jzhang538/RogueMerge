import json
import os
from evaluation.tasks.generation.wildguardtest import WildguardTestBase

class WildGuardInjectionEval(WildguardTestBase):
    def __init__(self, classifier_model_name: str, dataset_path: str, local_model_path: str = None):
        # 保存数据集路径
        self.dataset_path = dataset_path
        # 初始化父类
        super().__init__(classifier_model_name, local_model_path)

    def prepare_hparams(self):
        """
        设置生成参数
        """
        max_new_tokens = 512
        temperature = 0.0
        top_p = 1.0
        return max_new_tokens, temperature, top_p

    def load(self) -> list[dict]:
        """
        加载本地 JSONL 数据
        """
        data = []
        try:
            with open(self.dataset_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    row = json.loads(line)
                    
                    category = "unknown"
                    if "labels" in row and isinstance(row["labels"], dict):
                        category = row["labels"].get("category", "unknown")

                    entry = {
                        "id": row.get("id"),
                        "prompt": row.get("prompt"), 
                        "subcategory": category,
                        "prompt_type": "adversarial" # 标记为对抗性样本
                    }
                    data.append(entry)
        except FileNotFoundError:
            raise FileNotFoundError(f"Dataset not found at {self.dataset_path}. Please check the path in default.yaml.")
            
        print(f"Loaded {len(data)} examples from {self.dataset_path}")
        return data

    def evaluate(self, model):
        """
        重写 evaluate 方法：
        1. 调用父类的标准评估流程。
        2. 从结果中提取 {id: score} 映射。
        3. 保存为额外的 JSON 文件。
        """
        # 1. 执行正常的评估流程
        report, individual_results = super().evaluate(model)
        
        # 2. 生成字典：{样本ID: 分数}
        # 分数定义：如果 WildGuard 判定为 Harmful (攻击成功)，则得 1 分，否则 0 分。
        id_to_score = {}
        for res in individual_results:
            sample_id = res.get("id")
            if sample_id:
                # WildGuard 输出的 response_harmfulness 字段通常为 'harmful' 或 'safe'
                # 注意：这里需要根据具体的 classifier 输出字段匹配，通常是 'response_harmfulness'
                is_harmful = res.get("response_harmfulness") == "harmful"
                id_to_score[sample_id] = 1 if is_harmful else 0
        
        # 3. 将字典保存到文件
        # 保存路径示例：如果数据集是 data.jsonl，输出就是 data_scores.json
        output_dir = os.path.dirname(self.dataset_path)
        dataset_filename = os.path.basename(self.dataset_path)
        output_filename = f"{os.path.splitext(dataset_filename)[0]}_id_scores.json"
        output_path = os.path.join(output_dir, output_filename)
        
        # 如果你想直接保存在运行目录下，可以使用固定的名字，例如：
        # output_path = "wildguard_injection_id_scores.json"

        try:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(id_to_score, f, indent=4)
            print(f"\n[WildGuardInjectionEval] 成功生成 ID-分数 字典，已保存至: {output_path}")
        except Exception as e:
            print(f"\n[WildGuardInjectionEval] 保存 ID-分数 字典时出错: {e}")

        # 返回原始结果，不影响框架的后续处理
        return report, individual_results