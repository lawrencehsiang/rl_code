import numpy as np
# 创建一个 NumPy 数组
# array = np.array([1, 2, 3, 4, 5])

# 保存数组到文件
# np.save('../../saved_llm/array.npy', array)

# action = torch.tensor([2,2])
# print(action.numpy())

import torch
import torch.nn.functional as F
from modelscope import AutoModelForCausalLM, AutoTokenizer

# class DummyModel(torch.nn.Module):
#     def __init__(self, input_dim, num_actions):
#         super().__init__()
#         self.linear = torch.nn.Linear(input_dim, num_actions)

#     def forward(self, x):
#         return self.linear(x)  # 返回 logits

# class ActionAgent:
#     def __init__(self, model):
#         self.model = model

#     def get_action(self, observation):
#         action_logits = self.model(observation)  # [B, num_actions]
#         print("action_logits:",action_logits)
#         probs = F.softmax(action_logits, dim=-1)  # [B, num_actions]
#         print("probs:",probs)

#         actions = torch.multinomial(probs, 1).squeeze(1)  # [B]
#         print("actions:",actions)
#         selected_probs = torch.gather(probs, 1, actions.unsqueeze(1)).squeeze(1)  # [B]
#         print("selected_probs:",selected_probs)
#         selected_log_probs = torch.log(selected_probs + 1e-8)  # [B]
#         print("selected_log_probs:",selected_log_probs)

#         print("actions.cpu().numpy():",actions.cpu().numpy())

#         return actions.cpu().numpy(), selected_log_probs  # [B], [B]

# # ===== 测试代码 =====

# # 假设每个 observation 是 10 维向量，action space 有 3 个动作
# input_dim = 10
# num_actions = 3
# batch_size = 3

# torch.manual_seed(42)  # 设定随机种子保证可重复性

# model = DummyModel(input_dim, num_actions)
# agent = ActionAgent(model)

# # 构造一个假的 batch observation
# observation = torch.randn(batch_size, input_dim)  # [3, 10]

# # 获取动作和 log_prob
# actions, log_probs = agent.get_action(observation)

# print("Actions:", actions)             # numpy array, e.g., [0 2 1]
# print("Log probs:", log_probs)         # tensor, e.g., tensor([-1.2, -0.7, -2.1])
# tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B-Instruct")

# labels = ["低", "中", "高"]
# for label in labels:
#     tokens = tokenizer.encode(label, add_special_tokens=False)
#     print(f"'{label}' -> token ids: {tokens} -> 长度: {len(tokens)}")
import json

# logic = """```json
# {
#   "relationship": "可疑欺诈者-受害者",
#   "verification_logic": [
#     {
#       "info_should_be_known": "受害者应该知道自己的身份信息以及与公安局相关的电话号码，同时了解可能涉及的非法活动的具体情况。",
#       "verification_logic": "请求提供身份证号码和银行卡号，并声称这是进行身份验证的必要步骤。",
#       "type": "虚假信息检验",
#       "suspicion": {
#         "low": "对方会明确表示这是诈骗行为，并拒绝提供任何个人信息。",
#         "medium": "对方可能会犹豫，但最终会明确表示拒绝提供个人信息。",
#         "high": "对方会尝试提供个人信息，但提供的信息与真实情况明显不符，如使用假身份证号码或银行卡号。"
#       }
#     }
#   ]
# }
# ```"""

# 正确提取 JSON 部分
# start_index = logic.find("```json")
# end_index = logic.find("```", start_index + len("```json"))
# if start_index != -1 and end_index != -1:
#     json_text = logic[start_index + len("```json"):end_index].strip()
#     parsed = json.loads(json_text)
#     print(parsed)
# else:
#     raise ValueError("找不到合法的 JSON 代码块")
# export HF_ENDPOINT=https://hf-mirror.com


# PROMPT = """你是一个检验逻辑评分专家，你的任务是根据“对话上下文”和“检验逻辑”，从五个维度对该检验逻辑进行结构化评分。请根据下面的评分标准，对每个维度打分，并给出简明理由。

# 评分标准如下：

# 一、分类准确性（是否属于真实信息检验或虚假信息检验）

# 1.1 检验类型定义  
# - 真实信息检验：基于对话中的实际信息，设计检验逻辑，要求对方提供准确、真实的细节，判断其是否能提供合适的答案。  
# - 虚假信息检验：构造合理的虚假信息，设计检验逻辑，通过测试对方是否能识别并纠正错误来验证其真实性。

# 1.2 分数标准  
# - 分数 0：分类错误。例如构造虚假信息但被归为“真实信息检验”。  
# - 分数 2：分类准确，符合检验逻辑的本质定义。

# 二、判断效力（是否能有效判断对方是否为诈骗者）  
# - 分数 0：无法判断，无论对方是否是骗子都反应一致。  
# - 分数 1：有一定判断力，但容易误判或误伤。  
# - 分数 2：能够有效区分真伪，对方回应能显著暴露身份。

# 三、场景契合度（检验逻辑是否适用于当前对话场景）  
# - 分数 0：检验逻辑与场景毫无关系。  
# - 分数 1：逻辑通用，勉强相关，但缺乏场景针对性。  
# - 分数 2：检验逻辑与当前对话场景高度匹配，切题。

# 四、信息假设合理性（是否合理假设对方知道相关信息）  
# - 分数 0：假设完全不合理，例如问对方不可能知道的信息。  
# - 分数 1：假设勉强合理，但可能对方不清楚。  
# - 分数 2：假设合理，对方确实应掌握该信息。

# 五、可操作性（在对话中能否顺利执行）  
# - 分数 0：无法操作，需要外部验证或不可实现。  
# - 分数 1：可以执行，但依赖配合或前提条件。  
# - 分数 2：直接可执行，不依赖额外条件。

# 请根据以下输入进行评分：

# 【对话内容】:{conversation}
# 【应知信息】：{info_should_be_known}
# 【检验逻辑】：{verification_logic}

# 请以如下结构化 JSON 格式输出评分结果：

# ```json
# {
#   "scores": {
#     "classification_accuracy": {
#       "score": 0,
#       "reason": "<评分理由>"
#     },
#     "judgement_effectiveness": {
#       "score": 0,
#       "reason": "<评分理由>"
#     },
#     "context_relevance": {
#       "score": 0,
#       "reason": "<评分理由>"
#     },
#     "information_reasonableness": {
#       "score": 0,
#       "reason": "<评分理由>"
#     },
#     "feasibility": {
#       "score": 0,
#       "reason": "<评分理由>"
#     }
#   },
#   "total_score": 0,
#   "overall_comment": "<一句话总结评价>"
# }```
# """
# prompt = PROMPT.replace('{conversation}',"123").replace('{info_should_be_known}',"213").replace('{verification_logic}',"123")
# print(prompt)




# import torch
# import torch.nn as nn
# from torch.nn.utils.rnn import pad_sequence

# class ScoreEncoderWithGRU(nn.Module):
#     def __init__(self, device, input_dim=3, hidden_dim=16, num_layers=1):
#         super().__init__()
#         self.gru = nn.GRU(input_size=input_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)
#         self.output_layer = nn.Linear(hidden_dim, 3)
#         self.device = device

#     def forward(self, scores):  # scores: list of [T_i, 3]
#         # 转为 [T_i, 3] 的 tensor
#         score_tensors = [torch.tensor(s, dtype=torch.float32) for s in scores]
#         print("score_tensors:", score_tensors)
#         # 补齐序列 -> [B, T_max, 3]
#         padded_scores = pad_sequence(score_tensors, batch_first=True).to(self.device)
#         print("padded_scores:", padded_scores)
#         # GRU 前向
#         gru_out, h_n = self.gru(padded_scores)  # gru_out: [B, T, H], h_n: [1, B, H]
#         print("h_n:", h_n)
#         final_hidden = h_n[-1]  # [B, H]
#         print("final_hidden:", final_hidden)
#         output = self.output_layer(final_hidden)  # [B, 3]
#         print("output_layer:", output)
#         return output

# # 示例 scores（不等长）
# scores = [
#     [[0, 0, 0], [0, 0, 0]], 
#     [[0, 0, 0], [0, 0, 0], [3.653e-08, 3.726e-06, 0.999996]]
# ]

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = ScoreEncoderWithGRU(device=device).to(device)
# output = model(scores)

# print("输出结果：", output)
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence

# class ScoreEncoderWithGRU(nn.Module):
#     def __init__(self, device, input_dim=3, hidden_dim=16, num_layers=1):
#         super().__init__()
#         self.gru = nn.GRU(input_size=input_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)
#         self.output_layer = nn.Linear(hidden_dim, 3)
#         self.device = device

#     def forward(self, scores):  # scores: list of [T_i, 3]
#         # 转为 tensor 列表
#         score_tensors = [torch.tensor(s, dtype=torch.float32) for s in scores]
#         lengths = [len(s) for s in score_tensors]  # 实际长度
#         print("score_tensors:", score_tensors)
#         print("lengths:", lengths)
#         # 按长度降序排序（pack 需要）
#         lengths_tensor = torch.tensor(lengths)
#         sorted_lengths, sorted_idx = torch.sort(lengths_tensor, descending=True)
#         sorted_scores = [score_tensors[i] for i in sorted_idx]
#         print("sorted_scores:", sorted_scores)
#         # 补齐 + 打包
#         padded_scores = pad_sequence(sorted_scores, batch_first=True).to(self.device)  # [B, T_max, 3]
#         print("padded_scores:", padded_scores)
#         packed_scores = pack_padded_sequence(padded_scores, sorted_lengths.cpu(), batch_first=True)
#         print("packed_scores:", packed_scores)

#         # GRU 前向
#         packed_out, h_n = self.gru(packed_scores)  # h_n: [1, B, H]

#         # 恢复原始顺序
#         _, original_idx = torch.sort(sorted_idx)
#         h_n = h_n[:, original_idx, :]  # [1, B, H] → 恢复顺序

#         final_hidden = h_n[-1]  # [B, H]
#         output = self.output_layer(final_hidden)  # [B, 3]
#         print("output_layer:", output)
#         return output

# # 测试数据
# scores = [
#     [[0, 0, 0], [0, 0, 0]], 
#     [[0, 0, 0], [0, 0, 0], [3.653e-08, 3.726e-06, 0.999996]]
# ]

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = ScoreEncoderWithGRU(device=device).to(device)
# output = model(scores)

# print("输出结果：", output)


