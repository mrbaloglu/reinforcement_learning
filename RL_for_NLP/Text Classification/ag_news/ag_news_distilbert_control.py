"""import platform
root_path = "/Users/emrebaloglu/Documents/RL/basic_reinforcement_learning"

if platform.system() == "Windows":
    root_path = ""
    sep = '\\'
    print("Running on windows...")"""
from pathlib import Path 
import os
import sys
import re
import gc 

# Get the directory of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))
print("Current directory: ", current_dir)
dir = Path(__file__).parent
while not str(dir).endswith("basic_reinforcement_learning"):
    print(dir)
    dir = dir.parent
print("Parent directory: ", dir)
sys.path.append(os.path.abspath(dir))
# sys.path.append("/Users/emrebaloglu/Documents/RL/basic_reinforcement_learning")
# sys.path.append(os.path.dirname(os.path.realpath(__file__)))
"""import os
os.chdir("/Users/emrebaloglu/Documents/RL/basic_reinforcement_learning")"""

from datetime import datetime
import torch as th

#from NLP_utils.preprocessing import *
import NLP_utils.preprocessing as nlp_processing
from RL_for_NLP.text_environments import TextEnvClfControlForBertModels
from RL_for_NLP.text_reward_functions import calculate_stats_from_cm

from RL_for_NLP.text_data_pools import PartialReadingDataPoolWithTokens, PartialReadingDataPoolWithBertTokens
import reinforce_algorithm_utils as rl_monte_carlo
import policy_networks as pn

import torch as th
from torch.optim import Adam
from torchsummary import summary
import mlflow
from RL_for_NLP.text_reward_functions import calculate_stats_from_cm

import actor_critic_algortihm_utils as a2c_utils 

from stable_baselines3 import A2C, DQN, PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.dqn.policies import MlpPolicy
from stable_baselines3 import DQN

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score
import json
from collections import Counter
from tqdm import tqdm
from tqdm.notebook import tqdm
import mlflow

if __name__ == "__main__":

    
    
    print(dir)
    print(datetime.now())

    device = th.device("cuda" if th.cuda.is_available() else "cpu")
    print(f"Running on device: {device}")
    data = nlp_processing.openDfFromPickle("NLP_datasets/ag_news/ag_news_train_distilbert-base-uncased.pkl")

    pool = PartialReadingDataPoolWithBertTokens(data, "text", "label", 200, 20, mask = True)

    print(pool.possible_actions)
    env = TextEnvClfControlForBertModels(pool, 30522, int(1e+5), reward_fn="score", random_walk=True)

    print(env.action_space.actions, env.action_space._ix_to_action)
    print(env.clf_action_space.actions, env.clf_action_space._ix_to_action)
    print(env.n_action_space.actions, env.n_action_space._ix_to_action)
    print("="*40)


    MAX_CHUNK_LEN = env.pool.window_size
    FREEZE_EXTRACTOR = True
    print(f"Maximum length in chunks is {MAX_CHUNK_LEN}")

    extractor = a2c_utils.BertFeatureExtractor(max_len=MAX_CHUNK_LEN, freeze=FREEZE_EXTRACTOR)

    extractor_out_dim = extractor.out_dim

    clf_policy = a2c_utils.DenseActorCriticPolicy(extractor_out_dim, len(env.clf_action_space.actions), hidden_dim=10)
    stop_policy = a2c_utils.DenseActorCriticPolicy(extractor_out_dim, 2, hidden_dim=10)
    next_policy = a2c_utils.DenseActorCriticPolicy(extractor_out_dim, len(env.n_action_space.actions), hidden_dim=10)
    if FREEZE_EXTRACTOR:
        clf_optimizer = Adam(list(clf_policy.parameters()) + list(stop_policy.parameters()), lr=0.01, weight_decay=0.01)
        next_optimizer = Adam(list(next_policy.parameters()) + list(stop_policy.parameters()), lr=0.01, weight_decay=0.01)
    else:
        clf_optimizer = Adam(list(clf_policy.parameters()) + list(stop_policy.parameters()), lr=0.01, weight_decay=0.01)
        next_optimizer = Adam(list(extractor.parameters()) + list(next_policy.parameters()) + list(stop_policy.parameters()), lr=0.01, weight_decay=0.01)

    
    a2c = a2c_utils.ActorCriticAlgorithmControlBertModel(stop_policy, clf_policy, next_policy, extractor,
                         env, clf_optimizer, next_optimizer, device=device, gamma=1.)
    
    stats = None
    mlflow.set_experiment("actor-critic-w-distilbert-extractor")
    mlflow.start_run()
    # th.autograd.set_detect_anomaly(True)
    for _ in range(15):
        th.cuda.empty_cache()
        a2c.train_a2c(50, 100, log_interval=4)

        # a2c.device = th.device("cpu")
        stats, reward = a2c.eval_model(env)
        # a2c.device = th.device("cuda" if th.cuda.is_available() else "cpu")
        mlflow.log_metric("accuracy", stats["accuracy"])
        mlflow.log_metric("f1", stats["f1"])
        mlflow.log_metric("reward", reward)


    mlflow.pytorch.log_model(a2c.clf_policy, "a2c_distilbert_clf.pth")
    mlflow.pytorch.log_model(a2c.stop_policy, "a2c_distilbert_stop.pth")
    mlflow.pytorch.log_model(a2c.next_policy, "a2c_distilbert_next.pth")
    mlflow.end_run()
    # TODO TODO memory leak cuda + nan outputs problem