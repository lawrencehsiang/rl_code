# 这个文件定义了一个用于评估可疑度的类 ResponseQualityEvaluator
# 给定对话历史和当前使用的检验逻辑，输出一个分数，代表当前轮次的可疑度
import torch
import torch.nn.functional as F
from modelscope import AutoModelForCausalLM, AutoTokenizer

PROMPT = """你是一个对话系统中的可疑度评估器，任务是判断潜在诈骗者的回复是否具有欺诈倾向。你将获得以下信息：

- 用户的最新发言：{user_message}
- 潜在诈骗者的回复：{scammer_message}

请根据上述对话，判断该回复的可疑程度，仅从以下三个等级中选择一个进行输出：

- 低（回复可信、明确、没有欺诈迹象）
- 中（存在一定疑点，如含糊其辞、回避、语义模糊）
- 高（明显可疑，如答非所问、自相矛盾、无法验证）

请只输出以下三个词中的一个：“低”、“中”或“高”，不要输出任何解释或附加内容。
"""

class ResponseQualityEvaluator:
    def __init__(self,model,tokenizer,cache_dir='/home/chen/.cache'):
        self.model_name = "Qwen/Qwen2.5-3B-Instruct"
         # self.model = AutoModelForCausalLM.from_pretrained(
        #     self.model_name,
        #     torch_dtype="auto",
        #     device_map="auto",
        #     cache_dir=cache_dir
        # )
        # self.tokenizer = AutoTokenizer.from_pretrained(self.model_name,cache_dir=cache_dir)
        self.model=model
        self.tokenizer = tokenizer

    def evaluate(self, conversation,logic):
        prompt = PROMPT.format(user_message=conversation, scammer_message=logic)

        # 编码输入
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits[0, -1]  # 最后一个位置的 logits

        # 获取“低”、“中”、“高”的 token id
        choices = ["低", "中", "高"]
        token_ids = [self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(label)[0]) for label in choices]

        # 提取 logits 并 softmax
        selected_logits = logits[token_ids]
        probs = F.softmax(selected_logits, dim=0)

        # 输出结果
        return {
            "labels": choices,
            "probs": probs.tolist(),         # 例如 [0.1, 0.3, 0.6]
            "predicted": choices[probs.argmax().item()]
        }
    
    def get_label_probs(self,prompt, labels):
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.model.device)

        probs = []
        for label in labels:
            target_tokens = self.tokenizer.encode(label, add_special_tokens=False)
            
            if len(target_tokens) == 1:
                # 单 token，直接用 logits
                with torch.no_grad():
                    outputs = self.model(input_ids)
                    logits = outputs.logits[0, -1]
                    logit = logits[target_tokens[0]]
                    probs.append(logit.item())
            else:
                # 多 token，计算联合概率
                full_input = torch.cat([input_ids, torch.tensor([target_tokens], device=self.model.device)], dim=1)
                with torch.no_grad():
                    outputs = self.model(full_input)
                    logits = outputs.logits

                log_probs = []
                for i, token_id in enumerate(target_tokens):
                    logit = logits[0, input_ids.size(1) + i - 1]
                    log_prob = F.log_softmax(logit, dim=-1)[token_id]
                    log_probs.append(log_prob)

                total_log_prob = sum(log_probs)
                probs.append(total_log_prob.item())

        # softmax 归一化概率
        probs_tensor = torch.tensor(probs)
        final_probs = F.softmax(probs_tensor, dim=0).tolist()

        return {
            "labels": labels,
            "probs": final_probs,
            "predicted": labels[torch.tensor(final_probs).argmax().item()]
        }

    

