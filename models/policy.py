import torch
import torch.nn as nn
import torch
import transformers
from tqdm import tqdm
# from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Tuple
import torch.nn as nn
import numpy as np
from models.critic import DoubleCritic
from models.ScoreEncoder import ScoreEncoderWithGRU
from modelscope import AutoModelForCausalLM, AutoTokenizer


class Policy(torch.nn.Module):
    def __init__(self, device, accelerator, policy_lm = "Qwen/Qwen2.5-3B-Instruct", critic_lm = "roberta-base", 
                cache_dir = '~/.cache', do_sample = True, max_new_tokens = 32, use_bfloat16 = False, eos_str = '\n'):
        super().__init__()
        self.policy_lm = policy_lm
        if use_bfloat16:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.policy_lm,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                cache_dir=cache_dir
            ).to(device)
            # self.model = AutoModelForCausalLM.from_pretrained(policy_lm, cache_dir=cache_dir,
            #                                                   torch_dtype = torch.bfloat16).to(device)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.policy_lm,
                torch_dtype="auto",
                cache_dir=cache_dir
            ).to(device)
            # self.model = AutoModelForCausalLM.from_pretrained(policy_lm, cache_dir=cache_dir).to(device)
        

        # self.tokenizer = AutoTokenizer.from_pretrained(policy_lm, trust_remote_code=True, cache_dir=cache_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(self.policy_lm,trust_remote_code=True,cache_dir=cache_dir)
        self.tokenizer.truncation_side = 'left'
        # 将填充标记（padding token）设置为结束标记（end-of-sequence token），并让填充标记的 ID 等于结束标记的 ID。
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.device = device
        self.softmax = torch.nn.Softmax(dim= -1)
        self.do_sample = do_sample
        self.max_new_tokens = max_new_tokens
        self.eos_str = eos_str
        self.accelerator = accelerator
        self.score_encoder = ScoreEncoderWithGRU(device).to(device)
        self.policy_head = nn.Sequential(
            nn.Linear(2048 + 3, 128), 
            nn.ReLU(),
            nn.Linear(128,3)
        ).to(device)



    def forward(self,observation):
        conversations = [obs.conversation for obs in observation]
        suspicions = [obs.suspicion for obs in observation]
        prompts = self.create_prompt(conversations)
        score_states = self.score_encoder(suspicions)
        print(f"Shape of score_states: {score_states.shape}")
        model_inputs = self.tokenizer(prompts, padding=True, return_tensors='pt', max_length=512, truncation=True)
        model_inputs = {k: v.to(self.device) for k, v in model_inputs.items()}
        outputs = self.model(**model_inputs, output_hidden_states=True)
        
        hidden_states = outputs.hidden_states[-1]
        print(f"Shape of hidden_states: {hidden_states.shape}")
        attention_mask = model_inputs['attention_mask']
        masked_hidden = hidden_states * attention_mask.unsqueeze(-1)

        sum_hidden = masked_hidden.sum(dim=1)  # 对所有 token 向量求和
        lengths = attention_mask.sum(dim=1).unsqueeze(-1)  # 每个样本的有效 token 个数
        lm_states = sum_hidden / lengths  # [B, H]
        print(f"Shape of lm_states: {lm_states.shape}")

        # print("lm_states:",lm_states)
        combined_features = torch.cat([score_states, lm_states], dim=-1)
        print(f"Shape of combined_features: {combined_features.shape}")
        action_logits = self.policy_head(combined_features)  # [B, 3]
        return action_logits


    def create_prompt(self,conversation):
        prompts = []
    
        for conv in conversation:
            prompt = f"给你一段对话\n{conv}\n如果根据现有对话无法做出判断，选择0，表示继续下一轮对话，如果根据现有对话可以判断欺诈，选择1，表示结束对话，判断为欺诈，如果根据现有对话可以判断非欺诈，选择2，表示结束对话，判断为非欺诈\n请仅返回数字0、1或2\n"
            messages = [
                {"role": "system", "content": ""},
                {"role": "user", "content": prompt}
            ]
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            prompts.append(text)
        return prompts



