
import sys

sys.path.append("/Users/emrebaloglu/Documents/RL/basic_reinforcement_learning")
import os
os.chdir("/Users/emrebaloglu/Documents/RL/basic_reinforcement_learning")

import NLP_utils.preprocessing as nlp_processing
from RL_for_NLP.text_environments import TextEnvClfWithBertTokens, TextEnvClf, TextEnvClfForBertModels
from RL_for_NLP.text_reward_functions import calculate_stats_from_cm

from RL_for_NLP.text_data_pools import PartialReadingDataPoolWithTokens, PartialReadingDataPoolWithBertTokens
import reinforce_algorithm_utils as rl_monte_carlo
import policy_networks as pn

import torch as th
from torch.optim import Adam
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


data = nlp_processing.openDfFromPickle("NLP_datasets/ag_news/ag_news_train_distilbert-base-uncased.pkl")
pool = PartialReadingDataPoolWithBertTokens(data, "text", "label", 512, 50, mask = True)
env = TextEnvClfForBertModels(pool, 30522, int(1e+5), "score", True)


policy = a2c_utils.DistibertActorCriticPolicy(50, 4, dropout=0.)

optimizer = Adam(policy.parameters())
a2c = a2c_utils.ActorCriticAlgorithmBertModel(policy, env, optimizer, gamma=1.)


a2c.train_a2c(10, 10, log_interval=2)