import gym
from gym import spaces
from stable_baselines3.common.env_checker import check_env
import numpy as np
import pandas as pd
from typing import Union, Tuple
import pickle
import copy
import torch

import sys
sys.path.append("/Users/emrebaloglu/Documents/RL/basic_reinforcement_learning")

from NLP_utils import preprocessing as nlp_preprocessing
from NLP_utils import baseline_models as nlp_base_models
from NLP_utils import pytorch_datasets as nlp_datasets

from torch.utils.data import Dataset, DataLoader
from RL_for_NLP.text_data_pools import PartialReadingDataPoolWithWord2Vec, PartialReadingDataPoolWithBertTokens
from RL_for_NLP.text_action_space import ActionSpace
from RL_for_NLP.text_reward_functions import PartialReadingRewardFunctionF1, PartialReadingRewardFunctionAccuracy
from RL_for_NLP.observation import Observation


class TextEnvClf(gym.Env):
    
    def __init__(self, 
                data_pool: PartialReadingDataPoolWithWord2Vec, 
                max_time_steps: int, 
                reward_fn: str = "f1",
                random_walk: bool = False):
        super().__init__()
        assert reward_fn in ["f1", "accuracy", "precision", "recall"], \
            f"Reward functions needs to be one of 'f1', 'accuracy', 'precision', 'recall', got {reward_fn}."

        self.time_step = 0
        self.pool = data_pool
        self.random_walk = random_walk
        
        self.current_sample_ix = 0
        if self.random_walk:
            self.current_sample_ix = None
    
        self.current_observation = self.pool.create_episode(self.current_sample_ix)

        self.current_vecs = self.current_observation.get_sample_vecs()
        self.current_label = self.current_observation.get_label_enc()
        self.current_state_ix = 0
        self.current_state = self.current_vecs[self.current_state_ix]
        
        action_list = copy.deepcopy(self.pool.possible_actions)
        action_list.append("<next>")
        action_list.append("<previous>")
        self.action_space = ActionSpace(action_list)
        self.action_history = []
        self.prediction_history = []
        self.target_history = []
        self.sample_ix_history = []
        
        self.max_time_steps = max_time_steps

        if reward_fn == "f1":
            self.reward_function = PartialReadingRewardFunctionF1(self.pool.pos_label, self.pool.possible_actions)
        elif reward_fn == "accuracy":
            self.reward_function = PartialReadingRewardFunctionAccuracy(self.pool.possible_actions)
        else:
            raise NotImplementedError

        self._set_spaces()


    def step(self, action: int):
        
        action_str = self.action_space.ix_to_action(action)
        label_str = self.action_space.ix_to_action(self.current_label.item())
        step_reward = self.reward_function(action_str, label_str, self.prediction_history, self.target_history, self.time_step+1)
        
        if action_str in self.pool.possible_actions: # action_str == "good" or "bad":
            if self.current_sample_ix != None:
                self.current_sample_ix += 1
                self.current_sample_ix %= len(self.pool)

            self.current_observation = self.pool.create_episode(self.current_sample_ix)
            self.current_vecs = self.current_observation.get_sample_vecs() 
            self.current_label = self.current_observation.get_label_enc()
            self.prediction_history.append(action_str)
            self.target_history.append(label_str)
            self.sample_ix_history.append(self.current_sample_ix)
            self.current_state_ix = 0

        elif action_str == "<previous>":
            self.current_state_ix -= 1
        elif action_str == "<next>":
            self.current_state_ix += 1
            
        self.current_state_ix %= len(self.current_vecs)
        self.current_state = self.current_vecs[self.current_state_ix]
        self.action_history.append(action_str)
        

        
        self.time_step += 1
        done = (self.time_step > self.max_time_steps)
            
        
        dict_ = {"text": self.current_observation.get_sample_str(), "label": label_str, "action": action_str, "reward": step_reward}

        return self.current_state.astype(np.int32), step_reward, done, dict_ 

    def reset(self):
        if self.current_sample_ix != None:
            self.current_sample_ix += 1

        self.current_observation = self.pool.create_episode(self.current_sample_ix)
        self.current_vecs = self.current_observation.get_sample_vecs()
        self.current_label = self.current_observation.get_label_enc()
        self.current_state_ix = 0
        self.current_state = self.current_vecs[self.current_state_ix]
        self.time_step = 0
        self.action_history = []
        self.target_history = []
        self.prediction_history = []

        return self.current_state.astype(np.int32) # .detach().numpy()
    
    def get_prediction_history(self):
        return copy.deepcopy(self.prediction_history)
    def get_target_history(self):
        return copy.deepcopy(self.target_history)
    def get_sample_ix_history(self):
        return copy.deepcopy(self.sample_ix_history)


    def _set_spaces(self):
        low = np.full((self.pool.window_size, ), fill_value=-1)
        high = np.full((self.pool.window_size, ), fill_value=int(1e+6))
        self.observation_space = spaces.Box(low, high, dtype=np.int32)

class TextEnvClfBert(gym.Env):
    
    def __init__(self, 
                data_pool: PartialReadingDataPoolWithBertTokens,
                 max_time_steps: int, 
                 reward_fn: str = "f1",
                 random_walk: bool = False):
        super().__init__()

        assert reward_fn in ["f1", "accuracy", "precision", "recall"], \
            f"Reward functions needs to be one of 'f1', 'accuracy', 'precision', 'recall', got {reward_fn}."

        self.time_step = 0
        self.pool = data_pool
        self.random_walk = random_walk
        
        self.current_sample_ix = 0
        if self.random_walk:
            self.current_sample_ix = None
    
        self.current_observation = self.pool.create_episode(self.current_sample_ix)

        self.current_input_id_vecs = self.current_observation.get_sample_input_id_vecs()
        self.current_token_type_vecs = self.current_observation.get_sample_token_type_vecs()
        self.current_attn_mask_vecs = self.current_observation.get_sample_attn_mask_vecs()

        self.current_label = self.current_observation.get_label_enc()
        self.current_state_ix = 0
        self.current_state_input_id = self.current_input_id_vecs[self.current_state_ix]
        self.current_state_token_type = self.current_token_type_vecs[self.current_state_ix]
        self.current_state_attn_mask = self.current_attn_mask_vecs[self.current_state_ix]
        
        action_list = copy.deepcopy(self.pool.possible_actions)
        action_list.append("<next>")
        # action_list.append("<previous>")
        self.action_space = ActionSpace(action_list)
        self.action_history = []
        self.prediction_history = []
        self.target_history = []
        self.sample_ix_history = []
        
        self.max_time_steps = max_time_steps

        if reward_fn == "f1":
            self.reward_function = PartialReadingRewardFunctionF1(self.pool.pos_label, self.pool.possible_actions)
        elif reward_fn == "accuracy":
            self.reward_function = PartialReadingRewardFunctionAccuracy(self.pool.possible_actions)
        else:
            raise NotImplementedError
        self._set_spaces()


    def step(self, action: int):
        
        action_str = self.action_space.ix_to_action(action)
        label_str = self.action_space.ix_to_action(self.current_label.item())
        step_reward = self.reward_function(action_str, label_str, self.prediction_history, self.target_history, self.time_step+1)
        
        if action_str in self.pool.possible_actions: # action_str == "good" or "bad":
            if self.current_sample_ix != None:
                self.current_sample_ix += 1
                self.current_sample_ix %= len(self.pool)

            self.current_observation = self.pool.create_episode(self.current_sample_ix)
            self.current_input_id_vecs = self.current_observation.get_sample_input_id_vecs()
            self.current_token_type_vecs = self.current_observation.get_sample_token_type_vecs()
            self.current_attn_mask_vecs = self.current_observation.get_sample_attn_mask_vecs()
            self.current_label = self.current_observation.get_label_enc()
            self.prediction_history.append(action_str)
            self.target_history.append(label_str)
            self.sample_ix_history.append(self.current_sample_ix)
            self.current_state_ix = 0

        elif action_str == "<previous>":
            self.current_state_ix -= 1
        elif action_str == "<next>":
            self.current_state_ix += 1
            
        self.current_state_ix %= len(self.current_input_id_vecs)
        self.current_state_input_id = self.current_input_id_vecs[self.current_state_ix]
        self.current_state_token_type = self.current_token_type_vecs[self.current_state_ix]
        self.current_state_attn_mask = self.current_attn_mask_vecs[self.current_state_ix]
        self.action_history.append(action_str)
        

        
        self.time_step += 1
        done = (self.time_step > self.max_time_steps)
            
        
        dict_ = {"text": self.current_observation.get_sample_str(), "label": label_str, "action": action_str, "reward": step_reward}

        return self.current_state_input_id.astype(np.int32), step_reward, done, dict_ 

    def reset(self):
        if self.current_sample_ix != None:
            self.current_sample_ix += 1

        self.current_observation = self.pool.create_episode(self.current_sample_ix)
        self.current_input_id_vecs = self.current_observation.get_sample_input_id_vecs()
        self.current_token_type_vecs = self.current_observation.get_sample_token_type_vecs()
        self.current_attn_mask_vecs = self.current_observation.get_sample_attn_mask_vecs()
        self.current_label = self.current_observation.get_label_enc()
        self.current_state_ix = 0
        self.current_state_input_id = self.current_input_id_vecs[self.current_state_ix]
        self.current_state_token_type = self.current_token_type_vecs[self.current_state_ix]
        self.current_state_attn_mask = self.current_attn_mask_vecs[self.current_state_ix]
        self.time_step = 0
        self.action_history = []
        self.target_history = []
        self.prediction_history = []

        return self.current_state_input_id.astype(np.int32) # .detach().numpy()
    
    def get_prediction_history(self):
        return copy.deepcopy(self.prediction_history)
    def get_target_history(self):
        return copy.deepcopy(self.target_history)
    def get_sample_ix_history(self):
        return copy.deepcopy(self.sample_ix_history)


    def _set_spaces(self):
        low = np.full((self.pool.window_size, ), fill_value=-1)
        high = np.full((self.pool.window_size, ), fill_value=int(1e+6))
        self.observation_space = spaces.Box(low, high, dtype=np.int32)


if __name__ == "__main__":
    data = nlp_preprocessing.openDfFromPickle("NLP_datasets/RT_Polarity/rt-polarity-train-bert.pkl")
    pool = PartialReadingDataPoolWithBertTokens(data, "review", "label", "good", 8)
    env = TextEnvClfBert(pool, int(1e+5), False)
    check_env(env)
    print(env.step(1))
    print(env.step(2))
    print(env.step(1))
    print(env.current_state_input_id.shape, env.current_state_token_type.shape, env.current_state_attn_mask.shape)

    