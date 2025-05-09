import sys
import os

parent_dir = os.path.dirname(os.path.abspath(__file__)) + '/../'
sys.path.append(parent_dir)

import torch
import transformers
from tqdm import tqdm
from environment.scammer_detect import BatchedScammerDetectEnv
from models.DecisionAgent import DecisionAgent
from algorithm.offpolicy_train_loop import offpolicy_train_loop
from utils import colorful_print
import torch.nn as nn
import numpy as np 
import wandb
from omegaconf import DictConfig, OmegaConf
import os
import hydra
from accelerate import Accelerator
from datetime import timedelta
from accelerate import DistributedDataParallelKwargs, InitProcessGroupKwargs
transformers.logging.set_verbosity_error()

CONFIG_NAME = "scam"
@hydra.main(version_base=None, config_path="./config/", config_name=CONFIG_NAME)
def main(config: "DictConfig"):
    colorful_print(">>> Configuration file: "+CONFIG_NAME+"<<<", fg='blue')
    colorful_print(OmegaConf.to_yaml(config), fg='red')
    try:
        from huggingface_hub import login
        login(token=config.huggingface_token)
    except:
        print(">>> Huggingface token not found.")

    accelerator = Accelerator(InitProcessGroupKwargs(timeout=timedelta(18000)))
    device = accelerator.device


    env = BatchedScammerDetectEnv(bsize=config.batch_size)
    eval_env = env
    
    decode_f = lambda x:x
    # load decision model
   
    print(">>> Using Scam agent")
    agent = DecisionAgent(device=device, accelerator=accelerator, 
                        policy_lm=config.policy_lm, critic_lm=config.critic_lm,
                        cache_dir=config.cache_dir)
   
    tokenizer = agent.tokenizer
    if config.checkpoint_path is not None:
        state_dict = torch.load(config.checkpoint_path, map_location=device)['model_state_dict']
        agent.model.load_state_dict(state_dict)
    # agent = accelerator.prepare(agent)

    if config.use_wandb and accelerator.is_main_process:
        wandb.login(key=config.wandb_key)
        wandb.init(project=config.project_name, name=config.run_name, config=dict(config))

    offpolicy_train_loop(env = env,
                agent = agent,
                tokenizer = tokenizer,
                eval_env = eval_env,
                accelerator = accelerator,
                decode_f=decode_f,
                **config)



if __name__ == "__main__":
    main()
