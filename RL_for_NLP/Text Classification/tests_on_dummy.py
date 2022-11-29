import subprocess
subprocess.run("clear")

import numpy as np
import pandas as pd

import sys
sys.path.append("/Users/emrebaloglu/Documents/RL/basic_reinforcement_learning")
from RL_for_NLP.text_environments import TextEnvClfBert, TextEnvClf, SimpleSequentialEnv
from RL_for_NLP.text_data_pools import PartialReadingDataPoolWithWord2Vec, PartialReadingDataPoolWithBertTokens, SimpleSequentialDataPool
from RL_for_NLP.text_reward_functions import calculate_stats_from_cm
import NLP_utils.preprocessing as nlp_processing
import reinforce_algorithm_utils as rl_monte_carlo
import policy_networks as pn

import torch
from torch.optim import Adam
import mlflow

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



# declare some hyperparameters
WINDOW_SIZE = 10
MAX_STEPS = int(1e+5)
REWARD_FN = "score"

train_pool = SimpleSequentialDataPool(50000, 50, WINDOW_SIZE)
test_pool = SimpleSequentialDataPool(1000, 50, WINDOW_SIZE)
val_pool = SimpleSequentialDataPool(1000, 50, WINDOW_SIZE)


train_env = SimpleSequentialEnv(train_pool, MAX_STEPS, REWARD_FN)
val_env = SimpleSequentialEnv(val_pool, 50000, REWARD_FN)
test_env = SimpleSequentialEnv(test_pool, 50000, REWARD_FN)
print("All environments are created.")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}.")

def eval_model(model, env, total_timesteps=10000):
    done = False
    obs = env.reset()
    total_reward = 0.0
    actions = []
    seen_samples = 0
    for _ in tqdm(range(total_timesteps)):
        action, _states = model.predict(obs)
        action = action.item()
        obs, rewards, done, info = env.step(action)
        action = env.action_space.ix_to_action(action)
        if action in env.pool.possible_actions:
            seen_samples += 1
        if done:
            obs = env.reset()
        actions.append(action)
        total_reward += rewards
    print("---------------------------------------------");
    print(f"Total Steps and seen samples: {len(actions), seen_samples}")
    print(f"Total reward: {total_reward}")
    print(f"Stats:  {calculate_stats_from_cm(env.confusion_matrix)}")
    acts = list(Counter(actions).keys())
    freqs = list(Counter(actions).values())
    total = len(actions)
    print(f"Action stats --  {[{acts[ii]: freqs[ii]/total} for ii in range(len(acts))]}")
    print("---------------------------------------------")




# policy_kwargs = dict(
#     features_extractor_class=pn.CNN1DExtractor,
#     features_extractor_kwargs=dict(vocab_size = VOCAB_SIZE, embed_dim = 50, 
#                                 n_filter_list = [128, 64, 64, 32, 32, 16], kernel_size = 4, features_dim = 256),
# )
policy_kwargs = dict(
    features_extractor_class = pn.DummyNN,
    features_extractor_kwargs = dict(input_dim=WINDOW_SIZE, features_dim=10),
)
# policy_kwargs = dict(
#     features_extractor_class=pn.RNNExtractor,
#     features_extractor_kwargs=dict(vocab_size = VOCAB_SIZE, embed_dim = 5,
#                  rnn_type = "gru", rnn_hidden_size = 2, rnn_hidden_out = 2, rnn_bidirectional = True,
#                  features_dim = 3, units = 10),
# )
#policy_kwargs = dict(net_arch=[dict(pi=[512, 256, 64, 64], qf=[400, 300])])
# model = DQN("MlpPolicy", train_env, verbose=1, batch_size=10)
model = A2C(policy = "MlpPolicy",
            env = train_env,
            gae_lambda = 0.9,
            gamma = 0.99,
            learning_rate = 1e-3,
            max_grad_norm = 0.5,
            n_steps = 10,
            vf_coef = 0.4,
            ent_coef = 0.0,
            normalize_advantage=False,
            policy_kwargs=policy_kwargs,
            verbose=0, 
            use_rms_prop=True)
# model = PPO("MlpPolicy", train_env, policy_kwargs=policy_kwargs)


for i in range(int(25)):
    model.learn(total_timesteps=int(1e+3)*7, reset_num_timesteps=True, progress_bar=True)
    train_env.set_train_mode(False)
    print("======= On train env: ===============")
    eval_model(model, train_env)
    print("======= On val env: ===============")
    eval_model(model, val_env)
    train_env.set_train_mode(True)

