"""import subprocess
subprocess.run("clear")
"""
import numpy as np
import pandas as pd
import platform

root_path = "C:\\Users\\mrbal\\Documents\\NLP\\RL\\basic_reinforcement_learning"
info_path = ""
sep = '/'
if platform.system() == "Windows":
    info_path = root_path + "\\NLP_datasets\\RT_Polarity\\data_info_bert_windows.json"
    sep = '\\'
    print("Running on windows...")
else:
    root_path = "/Users/emrebaloglu/Documents/RL/basic_reinforcement_learning"
    info_path = root_path + "/NLP_datasets/RT_Polarity/data_info_bert.json"

import sys
sys.path.append(root_path)
from RL_for_NLP.text_environments import TextEnvClfBert, TextEnvClf
from RL_for_NLP.text_reward_functions import calculate_stats_from_cm

from RL_for_NLP.text_data_pools import PartialReadingDataPoolWithTokens, PartialReadingDataPoolWithBertTokens
import NLP_utils.preprocessing as nlp_processing
import reinforce_algorithm_utils as rl_monte_carlo
import policy_networks as pn

import torch
from torch.optim import Adam
import mlflow
from RL_for_NLP.text_reward_functions import calculate_stats_from_cm


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


with open(info_path, "r") as f:
    data_info = json.load(f)


data_train = nlp_processing.openDfFromPickle(data_info["path"] + sep + "rt-polarity-train-bert.pkl")




# declare some hyperparameters
WINDOW_SIZE = 16
MAX_STEPS = int(1e+5)
TRAIN_STEPS = int(1e+3)
VOCAB_SIZE = data_info["vocab_size"]
REWARD_FN = "score"
print(f"Vocab size: {VOCAB_SIZE}")

data_test = nlp_processing.openDfFromPickle(data_info["path"] + sep + "rt-polarity-test-bert.pkl")
data_val = nlp_processing.openDfFromPickle(data_info["path"] + sep + "rt-polarity-val-bert.pkl")

train_pool = PartialReadingDataPoolWithBertTokens(data_train, "review", "label", WINDOW_SIZE)
test_pool = PartialReadingDataPoolWithBertTokens(data_test, "review", "label", WINDOW_SIZE)
val_pool = PartialReadingDataPoolWithBertTokens(data_val, "review", "label", WINDOW_SIZE)

train_env_params = dict(data_pool=train_pool, max_time_steps=MAX_STEPS, reward_fn=REWARD_FN, random_walk=False, vocab_size=VOCAB_SIZE)
train_env = TextEnvClfBert(**train_env_params)
val_env = TextEnvClfBert(val_pool, 1000, reward_fn=REWARD_FN, vocab_size=VOCAB_SIZE)
test_env = TextEnvClfBert(test_pool, 1000, reward_fn=REWARD_FN, vocab_size=VOCAB_SIZE)
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





"""policy_kwargs = dict(
    features_extractor_class=pn.CNN1DExtractor,
    features_extractor_kwargs=dict(vocab_size = VOCAB_SIZE, embed_dim = 50, 
                                n_filter_list = [128, 64, 64, 32, 32, 16], kernel_size = 4, features_dim = 256),
)"""


policy_kwargs = dict(
    features_extractor_class=pn.RNNExtractor,
    features_extractor_kwargs=dict(vocab_size = VOCAB_SIZE, embed_dim = 5,
                 rnn_type = "gru", rnn_hidden_size = 2, rnn_hidden_out = 2, rnn_bidirectional = True,
                 features_dim = 10, units = 10),
)


model = DQN("MlpPolicy", train_env, policy_kwargs=policy_kwargs, verbose=1, batch_size=64, gamma=1.)
# model = A2C(policy = "MlpPolicy",
#             env = train_env,
#             gae_lambda = 0.9,
            # gamma = 1.,
#             learning_rate = 0.01,
#             max_grad_norm = 0.5,
#             n_steps = 1000,
#             vf_coef = 0.4,
#             ent_coef = 0.0,
#             policy_kwargs=policy_kwargs,
#             normalize_advantage=False,
#             verbose=0, 
#             use_rms_prop=False)
# model = PPO("MlpPolicy", train_env, policy_kwargs=policy_kwargs)

train_env.set_train_mode(False)
val_env.set_train_mode(False)
for i in range(int(15)):
    model.learn(total_timesteps=int(1e+4)*15, reset_num_timesteps=True, progress_bar=True)
    print("======= On train env: ===============")
    eval_model(model, train_env)
    print("======= On val env: ===============")
    eval_model(model, val_env)

model.save("nightly_trained_model")

