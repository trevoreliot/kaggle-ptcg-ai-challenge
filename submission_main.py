import os
import sys

# Suppress noisy warnings
os.environ["LITELLM_LOG"] = "ERROR"
os.environ["SUPPRESS_LITELLM_WARNINGS"] = "True"

# Prevent OpenMP thread contention in Kaggle's environment
import torch
torch.set_num_threads(1)

# Import the actual agent function which the Kaggle environment will call
from src.core.agent.agent import agent

# Kaggle environments looks for a function named `agent` or the last defined function
def agent_wrapper(obs, config=None):
    return agent(obs)
