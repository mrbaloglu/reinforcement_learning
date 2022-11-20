import subprocess
subprocess.run("clear")

import numpy as np
import pandas as pd

import sys
sys.path.append("/Users/emrebaloglu/Documents/RL/basic_reinforcement_learning")
from RL_for_NLP.text_environments import TextEnvClfBert, TextEnvClf
from RL_for_NLP.text_data_pools import PartialReadingDataPoolWithWord2Vec, PartialReadingDataPoolWithBertTokens
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


with open("/Users/emrebaloglu/Documents/RL/basic_reinforcement_learning/NLP_datasets/RT_Polarity/data_info_bert.json", "r") as f:
    data_info = json.load(f)

data_train = nlp_processing.openDfFromPickle(data_info["path"] + "/rt-polarity-train-bert.pkl")




# declare some hyperparameters
WINDOW_SIZE = 8
MAX_STEPS = int(1e+5)
VOCAB_SIZE = data_info["vocab_size"]
print(f"Vocab size: {VOCAB_SIZE}")

data_test = nlp_processing.openDfFromPickle(data_info["path"] + "/rt-polarity-test-bert.pkl")
data_val = nlp_processing.openDfFromPickle(data_info["path"] + "/rt-polarity-val-bert.pkl")

train_pool = PartialReadingDataPoolWithBertTokens(data_train, "review", "label", "good", WINDOW_SIZE)
test_pool = PartialReadingDataPoolWithBertTokens(data_test, "review", "label", "good", WINDOW_SIZE)
val_pool = PartialReadingDataPoolWithBertTokens(data_val, "review", "label", "good", WINDOW_SIZE)

train_env_params = dict(data_pool=train_pool, max_time_steps=MAX_STEPS, reward_fn="accuracy", random_walk=True)
train_env = TextEnvClfBert(**train_env_params)
val_env = TextEnvClfBert(val_pool, 1000, reward_fn="accuracy")
test_env = TextEnvClfBert(test_pool, 1000, reward_fn="accuracy")
print("All environments are created.")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}.")
def eval_model(model, env):
    done = False
    obs = env.reset()
    total_reward = 0.0
    actions = []
    while not done:
        action, _states = model.predict(obs)
        action = action.item()
        obs, rewards, done, info = env.step(action)
        actions.append(env.action_space.ix_to_action(action))
        total_reward += rewards
    print("---------------------------------------------")
    print(f"Predicted Label {actions}")
    print(f"Total Reward: {total_reward}")
    print(f"F1 Score: {f1_score(env.get_target_history(), env.get_prediction_history(), pos_label=env.pool.pos_label)}")
    print(f"Accuracy: {accuracy_score(env.get_target_history(), env.get_prediction_history())}")
    print("---------------------------------------------")



policy_kwargs = dict(
    features_extractor_class=pn.CNN1DExtractor,
    features_extractor_kwargs=dict(vocab_size = VOCAB_SIZE, embed_dim = 50, 
                                n_filter_list = [128, 64, 64, 32, 32, 16], kernel_size = 4, features_dim = 256),
)
# model = DQN("CnnPolicy", train_env, policy_kwargs=policy_kwargs, verbose=1, batch_size=64)
model = A2C(policy = "MlpPolicy",
            env = train_env,
            gae_lambda = 0.9,
            gamma = 0.99,
            learning_rate = 0.01,
            max_grad_norm = 0.5,
            n_steps = 10000,
            vf_coef = 0.4,
            ent_coef = 0.0,
            policy_kwargs=policy_kwargs,
            normalize_advantage=False,
            verbose=0, 
            use_rms_prop=False)
# model = PPO("MlpPolicy", train_env, policy_kwargs=policy_kwargs)


for i in range(int(5)):
    model.learn(total_timesteps=int(1e+4)*5, reset_num_timesteps=True, progress_bar=True)
    eval_model(model, val_env)


