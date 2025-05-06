import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from typing import Tuple
import torch.nn as nn
import numpy as np
from transformers import RobertaTokenizer, RobertaModel
from models.ScoreEncoder import ScoreEncoderWithGRU
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence

class DoubleCritic(torch.nn.Module):
   
    def __init__(self, device, accelerator, critic_lm, cache_dir, in_dim, out_dim):
        super(DoubleCritic, self).__init__()
        self.device = device
        self.accelerator = accelerator

        self.score_encoder = ScoreEncoderWithGRU(device).to(device)

        self.base_lm = AutoModel.from_pretrained(critic_lm, cache_dir=cache_dir).to(device)
        self.base_tokenizer = AutoTokenizer.from_pretrained(critic_lm, cache_dir=cache_dir)
       
        self.base_tokenizer.truncation_side = 'left'
       
        self.critic1 = nn.Sequential(nn.Linear(in_dim+3+3, in_dim),\
                                    nn.ReLU(),\
                                    nn.Linear(in_dim, in_dim),\
                                    nn.ReLU(),\
                                    nn.Linear(in_dim, out_dim)).to(device)
        self.critic2 = nn.Sequential(nn.Linear(in_dim+3+3, in_dim),\
                                    nn.ReLU(),\
                                    nn.Linear(in_dim, in_dim),\
                                    nn.ReLU(),\
                                    nn.Linear(in_dim, out_dim)).to(device)
        self.v_critic1 = nn.Sequential(nn.Linear(in_dim+3, in_dim),\
                                    nn.ReLU(),\
                                    nn.Linear(in_dim, in_dim),\
                                    nn.ReLU(),\
                                    nn.Linear(in_dim, out_dim)).to(device)
        self.v_critic2 = nn.Sequential(nn.Linear(in_dim+3, in_dim),\
                                    nn.ReLU(),\
                                    nn.Linear(in_dim, in_dim),\
                                    nn.ReLU(),\
                                    nn.Linear(in_dim, out_dim)).to(device)


    
    def forward(self, observation, action, detach_model=False):
        conversations = [obs.conversation for obs in observation]
        suspicions = [obs.suspicion for obs in observation]
        print("传入的suspicions:",suspicions)

        #suspicions = torch.tensor(suspicions, dtype=torch.float32).to(self.device)
        #print("转换成张量的suspicions:",suspicions)
        score_states = self.score_encoder(suspicions)
        
        obs_ids = self.base_tokenizer(conversations, padding = True, return_tensors='pt', max_length=512, truncation = True).to(self.device)
        if detach_model:
            with torch.no_grad():
                lm_states = self.base_lm(**obs_ids).pooler_output
        else:
            lm_states = self.base_lm(**obs_ids).pooler_output

        action = torch.tensor(action, dtype=torch.long).to(self.device)
        if detach_model:
            with torch.no_grad():
                action_one_hot = nn.functional.one_hot(action, num_classes=3).float()
        else:
            action_one_hot = nn.functional.one_hot(action, num_classes=3).float()
        q_states = torch.cat([score_states, lm_states, action_one_hot], dim=1)
        v_states =  torch.cat([score_states, lm_states], dim=1)
        return self.critic1(q_states), self.critic2(q_states), self.v_critic1(v_states), self.v_critic2(v_states)
    
