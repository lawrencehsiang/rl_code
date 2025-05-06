from torch.utils.data import Dataset
import numpy as np
# DummyDataset 类继承自 torch.utils.data.Dataset，用于封装一个数据缓冲区（buffer）。
# 这个类的主要作用是将数据缓冲区转换为 PyTorch 可以处理的数据集对象，方便后续使用 DataLoader 进行批量加载。
# __init__(self, buffer)：初始化方法，接收一个数据缓冲区 buffer 作为参数，并将其存储在类的实例属性 self.buffer 中。
# __len__(self)：返回数据缓冲区的长度，即数据集中样本的数量。
# __getitem__(self, idx)：根据索引 idx 从数据缓冲区中获取对应的样本并返回。
class DummyDataset(Dataset):
    def __init__(self, buffer):
        self.buffer = buffer

    def __len__(self):
        return len(self.buffer)

    def __getitem__(self, idx):
        return self.buffer[idx]


class Observation:
    def __init__(self, conversation, suspicion):
        self.conversation = conversation
        self.suspicion = suspicion

    def __repr__(self):
        return f"Observation(conversation='{self.conversation}', suspicion={self.suspicion})"

# 功能：ReplayBuffer 类用于存储智能体与环境交互产生的经验数据，这些数据包括观测值（observations）、动作（actions）、奖励（rewards）、下一个观测值（next_observations）、终止标志（dones）和蒙特卡罗回报（mc_returns）。
# 在强化学习中，经验回放是一种常用的技术，它可以提高数据的利用率，减少数据之间的相关性，从而提高算法的稳定性和收敛性。
# 初始化参数：
# batch_size：每次采样的样本数量，默认为 2。
# capacity：回放缓冲区的最大容量，默认为 10000。
class ReplayBuffer:
    def __init__(self, batch_size=2, capacity=10000):
        # 初始化经验回放缓冲区的最大容量
        self.max_size = capacity
        # 初始化当前缓冲区中存储的经验数量
        self.size = 0
        # 初始化存储观测（状态）的数组，初始为 None
        self.observations = None
        # 初始化存储奖励的数组，初始为 None
        self.rewards = None
        # 初始化存储下一个观测（下一个状态）的数组，初始为 None
        self.next_observations = None
        # 初始化存储是否完成标志的数组，初始为 None
        self.dones = None
        # 初始化每次采样的批量大小
        self.batch_size = batch_size
        # 初始化存储动作的数组，初始为 None
        self.actions = None
        # 初始化存储蒙特卡罗回报的数组，初始为 None
        self.mc_returns = None
 

    # 功能：从回放缓冲区中随机采样一批数据。
    # 参数解释：
    # batch_size：采样的样本数量，如果未提供，则使用初始化时指定的 batch_size。
    # 实现细节：
    # 生成 batch_size 个随机索引，这些索引的范围是从 0 到当前缓冲区的大小 self.size。
    # 使用这些随机索引从缓冲区中提取观测值、动作、奖励、下一个观测值、终止标志和蒙特卡罗回报，并将它们封装在一个字典中返回。
    def sample(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size
        # np.random.randint 是 NumPy 库中的一个函数，用于生成指定范围内的随机整数。
        # 0 是随机整数的下限（包含）。
        # self.size 是随机整数的上限（不包含），self.size 表示当前 ReplayBuffer 中存储的经验数据的数量。
        # size=(batch_size,) 表示生成的随机整数数组的形状，这里生成一个长度为 batch_size 的一维数组。
        # 例如，如果 self.size = 100，batch_size = 10，那么这部分代码可能会生成一个类似 [23, 45, 12, 78, 3, 56, 89, 1, 67, 90] 的数组。
        # 这个数组可以用于从 ReplayBuffer 的各个存储数组（如 self.observations、self.actions 等）中随机采样相应的经验数据。
        # {
        #     'observation': array(['obs3', 'obs6', 'obs8']),
        #     'action': array(['action3', 'action6', 'action8']),
        #     'reward': array([0.3, 0.6, 0.8]),
        #     'next_observation': array(['next_obs3', 'next_obs6', 'next_obs8']),
        #     'done': array([ True, False, False]),
        #     'mc_return': array([0.6, 1.2, 1.6])
        # }
        rand_indices = np.random.randint(0, self.size, size=(batch_size,)) % self.max_size
        return {
            "observation": self.observations[rand_indices],
            "action": self.actions[rand_indices],
            "reward": self.rewards[rand_indices],
            "next_observation": self.next_observations[rand_indices],
            "done": self.dones[rand_indices],
            "mc_return": self.mc_returns[rand_indices],
        }
    
    # 功能：返回当前回放缓冲区中存储的样本数量。
    def __len__(self):
        return self.size
    
    # 功能：将一个新的经验数据插入到回放缓冲区中。
    # 参数解释：
    # observation：当前的观测值。
    # action：智能体采取的动作。
    # reward：执行动作后获得的奖励。
    # next_observation：执行动作后环境的下一个观测值。
    # done：表示当前回合是否结束的标志。
    # mc_return：蒙特卡罗回报。
    # 实现细节：
    # 首先将 reward、mc_return 和 done 转换为 numpy 数组。
    # 如果缓冲区还没有初始化，则初始化观测值、动作、奖励、下一个观测值、终止标志和蒙特卡罗回报的数组。
    # 检查 reward 和 done 的形状是否为标量。
    # 将新的经验数据插入到缓冲区的当前位置（self.size % self.max_size）。
    # 增加缓冲区的大小 self.size。
    def insert(
        self,
        /,
        observation,
        action,
        reward: np.ndarray,
        next_observation,
        done: np.ndarray,
        mc_return,
        **kwargs
    ):
        """
        Insert a single transition into the replay buffer.

        Use like:
            replay_buffer.insert(
                observation=observation,
                action=action,
                reward=reward,
                next_observation=next_observation,
                done=done,
            )
        """
        if isinstance(reward, (float, int)):
            reward = np.array(reward)
        if isinstance(mc_return, (float, int)):
            mc_return = np.array(mc_return)
        if isinstance(done, bool):
            done = np.array(done)
        # print(next_observation)
        # if isinstance(prompt_actionaction, int):
        #     action = np.array(action, dtype=np.int64)

        # print("action::::",action)


        if self.observations is None:
            self.observations = np.array([None]*self.max_size, dtype = 'object')
            self.actions = np.empty((self.max_size, *action.shape), dtype=action.dtype)
            self.rewards = np.empty((self.max_size, *reward.shape), dtype=reward.dtype)
            self.next_observations = np.array([None]*self.max_size, dtype = 'object')
            self.dones = np.empty((self.max_size, *done.shape), dtype=done.dtype)
            self.mc_returns = np.empty((self.max_size, *mc_return.shape), dtype=mc_return.dtype)

        assert reward.shape == ()
        assert done.shape == ()

        self.observations[self.size % self.max_size] = observation
        self.actions[self.size % self.max_size] = action
        self.rewards[self.size % self.max_size] = reward
        self.next_observations[self.size % self.max_size] = next_observation
        self.dones[self.size % self.max_size] = done
        self.mc_returns[self.size % self.max_size] = mc_return

        self.size += 1