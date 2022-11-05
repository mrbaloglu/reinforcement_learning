import gym
from gym import spaces
import numpy as np
import pandas as pd
from typing import Union, Tuple
import pickle
import copy

import sys
sys.path.append("/Users/emrebaloglu/Documents/RL/basic_reinforcement_learning")

from NLP_utils import preprocessing as nlp_preprocessing
from NLP_utils import baseline_models as nlp_base_models
from NLP_utils import pytorch_datasets as nlp_datasets

from torch.utils.data import Dataset, DataLoader
from text_data_pools import RTDataPoolWithWord2Vec
from text_action_space import ActionSpace
from text_reward_functions import PartialReadingRewardFunctionF1


class TextEnvBinaryClf(gym.Env):
    
    def __init__(self, data_path: str, window_size: int, max_time_steps: int, use_bert_tokens: bool = False, print_dict: bool = False):
        super().__init__()
        if use_bert_tokens:
            pass
        else:
            self.pool = RTDataPoolWithWord2Vec(data_path, window_size, use_bert_tokens)
            self.loader = DataLoader(self.pool, batch_size=1, shuffle=True)
        
        self.print_dict = print_dict

        self.current_sample, self.current_label = next(iter(self.loader))
        self.current_state_ix = 0
        self.current_state = self.current_sample[self.current_state_ix]
        
        action_list = copy.deepcopy(self.pool.possible_actions)
        action_list.append("<next>")
        action_list.append("<reread>")
        action_list.append("<previous>")
        self.action_space = ActionSpace(action_list)
        self.action_history = []
        self.prediction_history = []
        self.target_history = []
        self.time_step = 0
        self.max_time_steps = max_time_steps

        self.reward_function = PartialReadingRewardFunctionF1()
        self._set_spaces()


    def step(self, action: int):
        
        action_str = self.action_space.ix_to_action(action)
        label_str = self.action_space.ix_to_action(self.current_label.item())
        if action_str == "good" or action_str == "bad":
            self.current_sample, self.current_label = next(iter(self.loader))
        elif action_str == "<previous>":
            self.current_state_ix -= 1
        elif action_str == "<next>":
            self.current_state_ix += 1
        self.current_state_ix = self.current_state_ix % len(self.current_sample)
        self.current_state = self.current_sample[self.current_state_ix]
        step_reward = self.reward_function(action_str, label_str, self.prediction_history, self.target_history)
        self.action_history.append(action_str)
        

        if action_str == "good" or action_str == "bad":
            self.prediction_history.append(action_str)
            self.target_history.append(label_str)

        self.time_step += 1
        done = False
        if self.time_step > self.max_time_steps:
            done = True
        
        dict_ = None
        if self.print_dict:
            dict_ = {"reward": step_reward}

        return self.current_state, step_reward, done, dict_


    def reset(self):
        sample, label = next(iter(self.loader)) 
        self.current_sample = sample
        self.current_label = label
        self.current_state = self.current_sample[0]
        self.time_step = 0
        self.action_history = []
        self.target_history = []
        self.prediction_history = []

        return self.current_state

    def _set_spaces(self):
        low = np.full(shape=(self.pool.window_size,), fill_value=-float('inf'), dtype=np.float32)
        high = np.full(shape=(self.pool.window_size,), fill_value=float('inf'), dtype=np.float32)
        self.observation_space = spaces.Box(low, high, dtype=np.float32)