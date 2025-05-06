import torch

from models.policy import Policy
import torch.nn.functional as F
from models.critic import DoubleCritic


class DecisionAgent(torch.nn.Module):
    def __init__(self, device, accelerator, policy_lm = "Qwen/Qwen2.5-3B-Instruct", critic_lm = "roberta-base", 
                cache_dir = '~/.cache'):
        super(DecisionAgent, self).__init__()
        self.model = Policy(device, accelerator, policy_lm = policy_lm, critic_lm = critic_lm, cache_dir = cache_dir)
        self.tokenizer = self.model.tokenizer
        self.critic = DoubleCritic(device, accelerator, critic_lm = critic_lm, cache_dir = cache_dir, in_dim = 768, out_dim = 1)  
        self.target_critic = DoubleCritic(device, accelerator, critic_lm = critic_lm, cache_dir = cache_dir, in_dim = 768, out_dim = 1) 
        self.soft_update_target_critic(1)
        self.accelerator = accelerator
       

    def prepare(self):
        self.model, self.critic, self.target_critic = self.accelerator.prepare(self.model, self.critic, self.target_critic)
    
    # observation_conversations, observation_scores
    def get_action(self, observation):
        action_logits = self.model(observation)
        # print("action_logits:",action_logits)
        probs = F.softmax(action_logits, dim=-1)       

        actions = torch.multinomial(probs,1).squeeze(1)

        selected_probs = torch.gather(probs,1,actions.unsqueeze(1)).squeeze(1)
        selected_log_probs = torch.log(selected_probs+1e-8)

        actions = actions.cpu().numpy()

        return actions, selected_log_probs  # 返回动作编号和对应概率
      


     # 该方法使用软更新策略更新目标评论网络的参数，通过一个平滑系数 tau 控制更新的速度。
    def soft_update_target_critic(self, tau):
        # for target_critic, critic in zip(self.target_critics, self.critics):
        for target_param, param in zip(
                self.target_critic.parameters(), self.critic.parameters()
            ):
                target_param.data.copy_(
                    target_param.data * (1.0 - tau) + param.data * tau
                )


    