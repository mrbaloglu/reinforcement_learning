import gym
import numpy as np
import pandas as pd
from typing import Union, Tuple
import pickle

import sys
sys.path.append("/Users/emrebaloglu/Documents/RL/basic_reinforcement_learning")

from NLP_utils import preprocessing as nlp_preprocessing
from NLP_utils import baseline_models as nlp_base_models
from NLP_utils import pytorch_datasets as nlp_datasets

from torch.utils.data import Dataset, DataLoader
from text_data_pools import RTDataPoolWithWord2Vec

def getData(data: Union[pd.DataFrame, str]) -> pd.DataFrame:
    if type(data) == pd.DataFrame:
        return data
    elif type(data) == str:
        if data.endswith(".csv"):
            return pd.read_csv(data)
        elif data.endswith(".pkl"):
            store_file = open("./rt-polarity/rt-processed-tokenized-padded.pkl", "rb")
            data = pickle.load(store_file)
            store_file.close()
            return data
        else:
            raise ValueError(f"Expected data file extension to be .csv or .pkl, got {data}")
    else:
        raise ValueError(f"Expected type of argument data as pd.DataFrame or str, got {type(data)}.")

class TextEnvBinaryClf(gym.Env):
    
    def __init__(self, data_path: str, window_size: int, use_bert_tokens: bool = False):
        
        if use_bert_tokens:
            pass
        else:
            self.pool = RTDataPoolWithWord2Vec(data_path, window_size, use_bert_tokens)
        
        



    def _step(self, action):
        """

        Parameters
        ----------
        action :

        Returns
        -------
        ob, reward, episode_over, info : tuple
            ob (object) :
                an environment-specific object representing your observation of
                the environment.
            reward (float) :
                amount of reward achieved by the previous action. The scale
                varies between environments, but the goal is always to increase
                your total reward.
            episode_over (bool) :
                whether it's time to reset the environment again. Most (but not
                all) tasks are divided up into well-defined episodes, and done
                being True indicates the episode has terminated. (For example,
                perhaps the pole tipped too far, or you lost your last life.)
            info (dict) :
                 diagnostic information useful for debugging. It can sometimes
                 be useful for learning (for example, it might contain the raw
                 probabilities behind the environment's last state change).
                 However, official evaluations of your agent are not allowed to
                 use this for learning.
        """
        self._take_action(action)
        self.status = self.env.step()
        reward = self._get_reward()
        ob = self.env.getState()
        episode_over = self.status != hfo_py.IN_GAME
        return ob, reward, episode_over, {}

    def _reset(self):
        pass

    def _render(self, mode='human', close=False):
        pass

    def _take_action(self, action):
        pass

    def _get_reward(self):
        """ Reward is given for XY. """
        if self.status == FOOBAR:
            return 1
        elif self.status == ABC:
            return self.somestate ** 2
        else:
            return 0