from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import LlamaForCausalLM, LlamaTokenizer
from transformers import AutoTokenizer, RobertaModel
import torch.nn as nn
import numpy as np
from data.utils import Observation


def add_trajectory_reward(trajectory):
    """
    add trajectory reward to the dict of each interaction
    """

    # 所以，这里就是计算终局奖励，并且为每个状态赋值终局奖励trajectory_reward
    trajectory_reward = np.sum([d["reward"] for d in trajectory])
    for d in trajectory:
        d.update({"trajectory_reward": trajectory_reward})
    return trajectory

# trajectory_rewards = array([[1, 2, 3]])
# gamma_row = array([[1.    , 0.95  , 0.9025]])
# gamma_matrix = array([[1.    , 0.95  , 0.9025],
#                       [0.    , 1.    , 0.95  ],
#                       [0.    , 0.    , 1.    ]])
# mc_returns = array([5.7575, 4.85  , 3.    ])
#[
#     {'observation': 'obs1', 'action': 'act1', 'reward': 1, 'done': False, 'mc_return': 5.7575},
#     {'observation': 'obs2', 'action': 'act2', 'reward': 2, 'done': False, 'mc_return': 4.85},
#     {'observation': 'obs3', 'action': 'act3', 'reward': 3, 'done': True, 'mc_return': 3.0}
# ]
def add_mc_return(trajectory, gamma = 0.95):
    """
    add trajectory reward to the dict of each interaction
    """
    # # 步骤 1: 提取轨迹中每个交互的奖励并转换为 numpy 数组
    # 从 trajectory 列表中的每个字典提取 reward 值，将其转换为 numpy 数组，并调整形状为 (1, n)，其中 n 是交互的数量
    trajectory_rewards = np.array([d["reward"] for d in trajectory]).reshape(1, -1)
    #  # 步骤 2: 计算折扣因子的累积乘积
    # 创建一个形状为 (1, n) 的全 1 数组，将其乘以折扣因子 gamma，然后计算累积乘积。
    gamma_row = np.cumprod(np.ones((1, trajectory_rewards.shape[1]))*gamma)
    # # 步骤 3: 计算折扣因子矩阵
    # 将 gamma_row 调整形状为 (1, n) 和 (n, 1)，然后进行除法运算，最后取上三角矩阵。
    gamma_matrix = np.triu(gamma_row.reshape(1, -1 )/ gamma_row.reshape(-1, 1))
    # # 步骤 4: 计算每个时间步的蒙特卡罗回报
    mc_returns = np.sum(trajectory_rewards*gamma_matrix, axis = 1)
    # 步骤 5: 将蒙特卡罗回报添加到每个交互的字典中
    for d, mc in zip(trajectory, mc_returns):
        d.update({"mc_return": mc})
    return trajectory



# 获取环境的批量大小 bsize。
# 初始化一个空列表 all_trajectories，用于存储所有轨迹。
# 使用 tqdm 循环 num_trajectories//bsize 次，每次处理一个批量的轨迹。
def batch_interact_environment(agent, tokenizer, env, num_trajectories,\
        post_f = lambda x: x, use_tqdm = True, decode_f = lambda x: x,
        env_idx = None):
    """
    in a bacthed way, interact with the environments  to get a list of trajectories
    [[{"observation":, "next_observation":, "reward":, "done":},...],...]
    post_f: function to add additional attributes to the trajectory
    """
    bsize = env.bsize
    print("bsize:",bsize)
    print("num_trajectories:",num_trajectories)
    all_trajectories = []
    for num_t in tqdm(range(num_trajectories//bsize), disable = not use_tqdm):
        print("num_t:",num_t)
        done = False
        # 初始化一个列表 trajectories，用于存储当前批量的轨迹。
        trajectories = [[] for _ in range(bsize)]
        # 调用 env.reset 方法重置环境，获取初始观测。
        # batch_obs_conversations是一个空的列表，["","",...]
        # return出来的batch_obs_scores 是一个列表，里面是每个环境的可疑度[[],[],...]
        batch_obs = env.reset()
        # 初始化一个布尔列表 batch_done，用于记录每个环境的完成状态。
        batch_done = [False,]*bsize
        steps = 0
        # 使用 while 循环，直到所有环境都完成。
        while not all(batch_done):
            steps += 1
            # print(f"Environment stpes {str(steps)}")
            # 调用 agent.get_action 方法根据当前观测生成动作。
            action,_ = agent.get_action(batch_obs)
            print("action:",action)
            # 调用 env.step 方法执行动作，获取下一个观测、奖励和完成状态。
            # decode_f 是一个函数，它的作用是对智能体（agent）生成的动作（action）进行解码。
            # 在强化学习中，智能体生成的动作可能是一种编码形式，而环境需要的是解码后的动作。
            # decode_f 就是用于将动作从编码形式转换为环境可以理解的形式。
            # decode_f 的默认值是 lambda x: x，这意味着如果没有提供 decode_f，则动作不会被解码，而是直接传递给环境。
            batch_return = env.step(decode_f(action))
            print("batch_return:",batch_return)
            for i,result in zip(range(bsize), batch_return):
                # 如果当前环境的result为None，代表已经结束，会直接跳过，None是在_step里赋值的
                if result is None:
                    continue
                next_obs_conversation,next_obs_scores, r, done = result
                
                ith_observation = batch_obs[i]
                i_next_observation = Observation(next_obs_conversation, next_obs_scores)
                trajectories[i].append({"observation":ith_observation, \
                                        "next_observation": i_next_observation, \
                                        "reward": r, \
                                        "done": done, \
                                        "action": action[i]})
              
                batch_obs[i] = i_next_observation
                batch_done[i] = done
            # obs = next_obs
        print("================================================")
        print(trajectories[0][-1]["next_observation"])
        print("================================================")
        all_trajectories += [post_f(add_mc_return(add_trajectory_reward(trajectory)))\
                              for trajectory in trajectories]
        # breakpoint()
        # trajectories.append(post_f(add_trajectory_reward(trajectory)))
    return all_trajectories


# 比如刚开始reset
# batch_obs_conversations是一个空的列表，["","",...]
# return出来的batch_obs_scores 是一个列表，里面是每个环境的可疑度[[],[],...]
# 然后get_action后，比如现在的action=[0,0,....0]
# 然后env.step(decode_f(action))
# 得到的batch_return 就是每个环境的self.conversation,self.scores, reward,self.done
# 然后组成{"observation":ith_observation, \
# "next_observation": i_next_observation, \
# "reward": r, \
# "done": done, \
# "action": action[i]}