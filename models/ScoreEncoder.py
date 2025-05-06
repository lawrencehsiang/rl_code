import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence


class ScoreEncoderWithGRU(nn.Module):
    def __init__(self, device,input_dim=3, hidden_dim=16, num_layers=1):
        super().__init__()
        self.gru = nn.GRU(input_size=input_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)
        self.output_layer = nn.Linear(hidden_dim, 3)  # 最终保留[low, mid, high]结构
        self.device = device

    def forward(self, scores):  # scores: list of [T_i, 3]
        # 转为 tensor 列表
        score_tensors = [torch.tensor(s, dtype=torch.float32) for s in scores]
        lengths = [len(s) for s in score_tensors]  # 实际长度
        # 按长度降序排序（pack 需要）
        lengths_tensor = torch.tensor(lengths)
        sorted_lengths, sorted_idx = torch.sort(lengths_tensor, descending=True)
        sorted_scores = [score_tensors[i] for i in sorted_idx]
        # 补齐 + 打包
        padded_scores = pad_sequence(sorted_scores, batch_first=True).to(self.device)  # [B, T_max, 3]
        packed_scores = pack_padded_sequence(padded_scores, sorted_lengths.cpu(), batch_first=True)
        # GRU 前向
        packed_out, h_n = self.gru(packed_scores)  # h_n: [1, B, H]

        # 恢复原始顺序
        _, original_idx = torch.sort(sorted_idx)
        h_n = h_n[:, original_idx, :]  # [1, B, H] → 恢复顺序

        final_hidden = h_n[-1]  # [B, H]
        output = self.output_layer(final_hidden)  # [B, 3]
        return output
