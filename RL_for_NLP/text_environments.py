import gym
from gym import spaces
from stable_baselines3.common.env_checker import check_env
import numpy as np
import pandas as pd
from typing import Union, Tuple, List
import pickle
import copy
import torch

import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from NLP_utils import preprocessing as nlp_preprocessing
from NLP_utils import baseline_models as nlp_base_models
from NLP_utils import pytorch_datasets as nlp_datasets

import intrinsic_text_value_models as itv_models

from torch.utils.data import Dataset, DataLoader
from RL_for_NLP.text_data_pools import PartialReadingDataPoolWithTokens, PartialReadingDataPoolWithBertTokens, SimpleSequentialDataPool, PartialReadingDataPool
from RL_for_NLP.text_action_space import ActionSpace
from RL_for_NLP.text_reward_functions import PartialReadingRewardF1, PartialReadingRewardAccuracy, PartialReadingRewardPrecision, PartialReadingRewardRecall, PartialReadingRewardScore
from RL_for_NLP.observation import Observation

import torch as th
import torch.nn as nn

from abc import ABC, abstractmethod


class BaseTextEnvClf(gym.Env, ABC):
    def __init__(self, data_pool: PartialReadingDataPool, vocab_size: int, max_time_steps: int, 
                 reward_fn: str = "score",  random_walk: bool = False):
        """Constructor for text environments for classification. All environments will inherit from this abstract class.

        Args:
            data_pool (PartialReadingDataPool): Pool for the textual data.
            vocab_size (int): Number of distinct words in the dataset used.
            max_time_steps (int): Maximum time steps for the agent.
            reward_fn (str, optional): Reward function of the environment.
                                       Must be one of "f1", "accuracy", "precision", "recall", "score". Defaults to "score".
            random_walk (bool, optional): Whether to select samples randomly. Defaults to False.
        """
        assert reward_fn in ["f1", "accuracy", "precision", "recall", "score"], \
            f"Reward functions needs to be one of 'f1', 'accuracy', 'precision', 'recall', 'score', got {reward_fn}."

        super().__init__()
        self.time_step = 0
        self.pool = data_pool
        self.random_walk = random_walk
        self.vocab_size = vocab_size
        action_list = copy.deepcopy(self.pool.possible_actions)
        action_list.append("<next>")
        # action_list.append("<previous>")
        self.action_space = ActionSpace(action_list)
        self.confusion_matrix = np.zeros((len(self.pool.possible_actions), len(self.pool.possible_actions)))
        
        self.current_sample_ix = 0
        if self.random_walk:
            self.current_sample_ix = None
    
        self.current_observation = self.pool.create_episode(self.current_sample_ix)
        

        self.max_time_steps = max_time_steps

        if reward_fn == "f1":
            self.reward_function = PartialReadingRewardF1(self.pool.possible_actions)
        elif reward_fn == "accuracy":
            self.reward_function = PartialReadingRewardAccuracy(self.pool.possible_actions)
        elif reward_fn == "precision":
            self.reward_function = PartialReadingRewardPrecision(self.pool.possible_actions)
        elif reward_fn == "recall":
            self.reward_function = PartialReadingRewardRecall(self.pool.possible_actions)
        else: 
            self.reward_function = PartialReadingRewardScore(self.pool.possible_actions)

        self.current_state_ix = 0
        self.last_reward = 0

    
    @abstractmethod 
    def update_current_state(self):
        raise NotImplementedError
    
    @abstractmethod
    def _set_spaces(self):
        raise NotImplementedError
    

    def calculate_reward(self, action: int) -> float: 
        """calculate the reward and update confusion matrix, given the action.

        Args:
            action (int): Action taken.

        Returns:
            float: Reward as a feedback for the action.
        """
        action_str = self.action_space.ix_to_action(action)
        label_str = self.action_space.ix_to_action(self.current_observation.get_label_enc().item())
        reward, self.confusion_matrix = self.reward_function(action_str, label_str, self.confusion_matrix, 1)
        step_reward = reward
        if not isinstance(self.reward_function, PartialReadingRewardScore):
            step_reward = reward - self.last_reward
        self.last_reward = reward

        return step_reward
    
    @abstractmethod
    def step(self, action: int) -> Tuple[float, bool, dict]:
        """As the current state update method varies on different envs, 
        this should be overriden in subclass methods.

        Args:
            action (int): Action taken.

        Returns:
            Tuple[float, bool, dict]: reward, done, info (state index is also updated)
        """

        step_reward = self.calculate_reward(action)
        action_str = self.action_space.ix_to_action(action)
        label_str = self.action_space.ix_to_action(self.current_observation.get_label_enc().item())

        if action_str in self.pool.possible_actions: # action_str == "good" or "bad":
            if self.current_sample_ix != None:
                self.current_sample_ix += 1
                self.current_sample_ix %=  len(self.pool)

            self.current_observation = self.pool.create_episode(self.current_sample_ix)
            self.current_state_ix = 0

        # elif action_str == "<previous>":
        #     self.current_state_ix -= 1
        elif action_str == "<next>":
            self.current_state_ix += 1
            
        self.current_state_ix %=  self.n_sentences_in_obs
        self.time_step += 1
        done = (self.time_step > self.max_time_steps)
            
        
        dict_ = {"text": self.current_observation.get_sample_str(), "label": label_str, "action": action_str, "reward": step_reward}

        return step_reward, done, dict_ 
    
    @abstractmethod
    def reset(self):
        """Class params are reset but as the current state update varies on different envs, 
        this should be overriden in subclass methods.
        """
        if self.current_sample_ix != None:
            self.current_sample_ix = 0

        self.current_observation = self.pool.create_episode(self.current_sample_ix)
        self.current_state_ix = 0
        self.time_step = 0
        self.last_reward = 0
    
class TextEnvClfControl(BaseTextEnvClf):
    def __init__(self, data_pool: PartialReadingDataPoolWithTokens, vocab_size: int,  max_time_steps: int,
                 clf_model: nn.Module, stop_model: nn.Module, next_model: nn.Module,
                 reward_fn: str = "f1", random_walk: bool = False) -> None:
        super().__init__(data_pool, vocab_size, max_time_steps, reward_fn, random_walk)
        self.clf_action_space = ActionSpace(copy.deepcopy(data_pool.possible_actions)) # list of class labels in data
        self.n_action_space = ActionSpace([f"<next_{ii}>" for ii in range(max_time_steps+1)]) # action to go to next chunks (next_0: reread, next_1: read next chunk, next_2: read chunk two next, ...)
        self.action_space = ActionSpace(["<stop>", "<continue>"]) # make a classification prediction or continue reading

        self.clf_model = clf_model
        self.stop_model = stop_model
        self.next_model = next_model

        self.current_state = self.update_current_state()
        self.n_sentences_in_obs = len(self.current_observation.get_sample_vecs())
        self._set_spaces()
    
    # create/update the current state that the agent will observe, using current observation and state index
    def update_current_state(self) -> np.ndarray:
        current_vecs = self.current_observation.get_sample_vecs()
       
        return current_vecs[self.current_state_ix].astype(int)
    
    def _set_spaces(self):
       self.observation_space = spaces.Box(0, self.vocab_size, shape=(self.pool.window_size, ), dtype=int) 
    
    def calculate_reward(self, action: str) -> float:
        """calculate the reward and update confusion matrix, given the action.

        Args:
            action (str): Action taken.

        Returns:
            float: Reward as a feedback for the action.
        """
        if action in self.action_space.actions: # <stop> or <continue>
            return 0.
        elif action in self.n_action_space.actions:
            return -1. # TODO FLOP calculation here 
        else:
            label_str = self.action_space.ix_to_action(self.current_observation.get_label_enc().item())
            reward, self.confusion_matrix = self.reward_function(action, label_str, self.confusion_matrix, 1)
            step_reward = reward
            if not isinstance(self.reward_function, PartialReadingRewardScore):
                step_reward = reward - self.last_reward
            self.last_reward = reward

            return step_reward

    def step(self, action: str) -> Tuple[np.ndarray, float, bool, dict]: 
        if action == "<stop>":
            # go to classifier actions
            pass
        elif action == "<continue>":
            # go to next chunk reading actions
            pass
        elif action in self.n_action_space.actions:
            # action = <next_n> where n is the skip size
            self.current_state_ix += self.n_action_space.action_to_ix(action)
            self.time_step += 1
            
        elif action in self.clf_action_space.actions:
            if self.current_sample_ix != None:
                self.current_sample_ix += 1
                self.current_sample_ix %=  len(self.pool)

            self.current_state_ix = 0
            self.current_observation = self.pool.create_episode(self.current_sample_ix)
            self.current_state = self.update_current_state()

        label_str = self.clf_action_space.ix_to_action(self.current_observation.get_label_enc().item())
        done = (self.time_step > self.max_time_steps)
        step_reward = self.calculate_reward(action)
        info = {"text": self.current_observation.get_sample_str(), "label": label_str, "action": action, "reward": step_reward}

        return self.current_state, step_reward, done, info
    
    def reset(self) -> np.ndarray:
        super().reset()
        self.current_state = self.update_current_state()
        return self.current_state_ix
    

class TextEnvClfControlForBertModels(BaseTextEnvClf):
    def __init__(self, data_pool: PartialReadingDataPoolWithBertTokens, vocab_size: int,  max_time_steps: int,
                 clf_model: nn.Module, stop_model: nn.Module, next_model: nn.Module,
                 reward_fn: str = "f1", random_walk: bool = False) -> None:
        super().__init__(data_pool, vocab_size, max_time_steps, reward_fn, random_walk)
        self.clf_action_space = ActionSpace(copy.deepcopy(data_pool.possible_actions)) # list of class labels in data
        self.n_action_space = ActionSpace([f"<next_{ii}>" for ii in range(max_time_steps+1)]) # action to go to next chunks (next_0: reread, next_1: read next chunk, next_2: read chunk two next, ...)
        self.action_space = ActionSpace(["<stop>", "<continue>"]) # make a classification prediction or continue reading

        self.clf_model = clf_model
        self.stop_model = stop_model
        self.next_model = next_model

        self.current_state = self.update_current_state()
        self.n_sentences_in_obs = len(self.current_observation.get_sample_vecs())
        self._set_spaces()
    
    # create/update the current state that the agent will observe, using current observation and state index
    def update_current_state(self):
        current_input_id_vecs = self.current_observation.get_sample_input_id_vecs()
        current_attn_mask_vecs = self.current_observation.get_sample_attn_mask_vecs()

        self.current_label = self.current_observation.get_label_enc()
        # self.current_state_input_id = self.current_input_id_vecs[self.current_state_ix]
        # self.current_state_attn_mask = self.current_attn_mask_vecs[self.current_state_ix]
        return {"input_id": current_input_id_vecs[self.current_state_ix].astype(int), 
                "attn_mask": current_attn_mask_vecs[self.current_state_ix].astype(int)}
    
    def _set_spaces(self):
        self.observation_space = spaces.Dict(
            {
                "input_id": spaces.Box(0, self.vocab_size, shape=(self.pool.window_size, ), dtype=int),
                "attn_mask": spaces.Box(0, 2, shape=(self.pool.window_size, ), dtype=int) 
            }
        )
    def calculate_reward(self, action: str) -> float:
        """calculate the reward and update confusion matrix, given the action.

        Args:
            action (str): Action taken.

        Returns:
            float: Reward as a feedback for the action.
        """
        if action in self.action_space.actions: # <stop> or <continue>
            return 0.
        elif action in self.n_action_space.actions:
            return -1. # TODO FLOP calculation here 
        else:
            label_str = self.action_space.ix_to_action(self.current_observation.get_label_enc().item())
            reward, self.confusion_matrix = self.reward_function(action, label_str, self.confusion_matrix, 1)
            step_reward = reward
            if not isinstance(self.reward_function, PartialReadingRewardScore):
                step_reward = reward - self.last_reward
            self.last_reward = reward

            return step_reward

    def step(self, action: str) -> Tuple[np.ndarray, float, bool, dict]: 
        if action == "<stop>":
            # go to classifier actions
            pass
        elif action == "<continue>":
            # go to next chunk reading actions
            pass
        elif action in self.n_action_space.actions:
            # action = <next_n> where n is the skip size
            self.current_state_ix += self.n_action_space.action_to_ix(action)
            self.time_step += 1
            
        elif action in self.clf_action_space.actions:
            if self.current_sample_ix != None:
                self.current_sample_ix += 1
                self.current_sample_ix %=  len(self.pool)

            self.current_state_ix = 0
            self.current_observation = self.pool.create_episode(self.current_sample_ix)
            self.current_state = self.update_current_state()

        label_str = self.clf_action_space.ix_to_action(self.current_observation.get_label_enc().item())
        done = (self.time_step > self.max_time_steps)
        step_reward = self.calculate_reward(action)
        info = {"text": self.current_observation.get_sample_str(), "label": label_str, "action": action, "reward": step_reward}

        return self.current_state, step_reward, done, info
    
    def reset(self) -> np.ndarray:
        super().reset()
        self.current_state = self.update_current_state()
        return self.current_state_ix
    

     

class TextEnvClf(BaseTextEnvClf):
    
    def __init__(self, data_pool: PartialReadingDataPoolWithTokens, vocab_size: int, max_time_steps: int, 
                 reward_fn: str = "f1", random_walk: bool = False):
        """Constructor for basic tokenized text environment for classification. 

        Args:
            data_pool (PartialReadingDataPoolWithTokens): Pool for the textual data.
            vocab_size (int): Number of distinct words in the dataset used.
            max_time_steps (int): Maximum time steps for the agent.
            reward_fn (str, optional): Reward function of the environment.
                                       Must be one of "f1", "accuracy", "precision", "recall", "score". Defaults to "f1".
            random_walk (bool, optional): Whether to select samples randomly. Defaults to False.
        """
        super().__init__(data_pool, vocab_size, max_time_steps, reward_fn, random_walk)
        self.current_state = self.update_current_state()
        self.n_sentences_in_obs = len(self.current_observation.get_sample_vecs())
        self._set_spaces()
        
    # create/update the current state that the agent will observe, using current observation and state index
    def update_current_state(self) -> np.ndarray:
        current_vecs = self.current_observation.get_sample_vecs()
       
        return current_vecs[self.current_state_ix].astype(int)
    
    def _set_spaces(self):
       self.observation_space = spaces.Box(0, self.vocab_size, shape=(self.pool.window_size, ), dtype=int) 
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        reward, done, info =  super().step(action)
        self.current_state = self.update_current_state()
        return self.current_state, reward, done, info
    
    def reset(self) -> np.ndarray:
        super().reset()
        self.current_state = self.update_current_state()
        return self.current_state
    
    # TODO burada observation ile alakalı bir sorun olabilir burdan devam et: https://www.gymlibrary.dev/content/environment_creation/ !!!
    """def _set_spaces(self):
        # low = np.full((self.pool.window_size, ), fill_value=-1)
        # high = np.full((self.pool.window_size, ), fill_value=)
        # self.observation_space = spaces.Box(low, high, dtype=np.int32)
        # self.observation_space = spaces.Dict(
        #     {
        #         "current": spaces.Box(0, self.pool.vocab_size, shape=(self.pool.window_size,), dtype=int),
        #         "next": spaces.Box(0, self.pool.vocab_size, shape=(self.pool.window_size), dtype=int),
        #     }
        # )
        self.observation_space = spaces.Box(0, self.pool.vocab_size, shape=(self.pool.window_size, ), dtype=int)"""

class TextEnvClfWithBertTokens(BaseTextEnvClf):
    
    def __init__(self, data_pool: PartialReadingDataPool, vocab_size: int, max_time_steps: int, 
                 reward_fn: str = "f1", max_skip_steps: int = 3, random_walk: bool = False):
        """Constructor for pretrained BERT tokenized text environment (only tokens) for classification. 

            Args:
                data_pool (PartialReadingDataPoolWithBertTokens): Pool for the textual data. (mask parameter should be False)
                vocab_size (int): Number of distinct words in the dataset used.
                max_time_steps (int): Maximum time steps for the agent.
                reward_fn (str, optional): Reward function of the environment.
                                        Must be one of "f1", "accuracy", "precision", "recall", "score". Defaults to "f1".
                random_walk (bool, optional): Whether to select samples randomly. Defaults to False.
        """
        super().__init__(data_pool, vocab_size, max_time_steps, reward_fn, random_walk)
        
        assert self.pool.mask == False, \
            "The mask parameter of the data pool must be False."

        self.current_state = self.update_current_state()
        self.n_sentences_in_obs = len(self.current_observation.get_sample_input_id_vecs())
        self._set_spaces()
        print("---- Inside constructor ---------")
        print("State shape/type: ", self.current_state.shape, type(self.current_state))
        print("------------- End Constructor-------------")

    
    def update_current_state(self):
        current_input_id_vecs = self.current_observation.get_sample_input_id_vecs()

        self.current_label = self.current_observation.get_label_enc()
        # self.current_state_input_id = self.current_input_id_vecs[self.current_state_ix]
        # self.current_state_attn_mask = self.current_attn_mask_vecs[self.current_state_ix]
        return current_input_id_vecs[self.current_state_ix].astype(int)
    
    def _set_spaces(self):
        self.observation_space = spaces.Box(0, self.vocab_size, shape=(self.pool.window_size, ), dtype=int)
    

    def step(self, action: int):
        reward, done, info = super().step(action)
        self.current_state = self.update_current_state()

        return self.current_state, reward, done, info 

    def reset(self):
        super().reset()
        self.current_state = self.update_current_state()
        return self.current_state
    
    def __len__(self) -> int:
        return len(self.pool)


class TextEnvClfForBertModels(BaseTextEnvClf):
    
    def __init__(self, data_pool: PartialReadingDataPoolWithBertTokens, vocab_size: int, max_time_steps: int, reward_fn: str = "f1", random_walk: bool = False):
        """Constructor for basic tokenized text environment for classification. 

        Args:
            data_pool (PartialReadingDataPoolWithBertTokens): Pool for the textual data.
            vocab_size (int): Number of distinct words in the dataset used.
            max_time_steps (int): Maximum time steps for the agent.
            reward_fn (str, optional): Reward function of the environment.
                                       Must be one of "f1", "accuracy", "precision", "recall", "score". Defaults to "f1".
            random_walk (bool, optional): Whether to select samples randomly. Defaults to False.
        """
        super().__init__(data_pool, vocab_size, max_time_steps, reward_fn, random_walk)
        
        assert self.pool.mask == True, \
            "The mask parameter of the data pool must be True."

        self.current_state = self.update_current_state()
        self.n_sentences_in_obs = len(self.current_observation.get_sample_input_id_vecs())
        self._set_spaces()
        print("---- Inside constructor ---------")
        print("Input id shape/type: ", self.current_state["input_id"].shape, type(self.current_state["input_id"]))
        print("Attn mask shape/type: ", self.current_state["attn_mask"].shape, type(self.current_state["attn_mask"]))
        print("------------- End Constructor-------------")

    
    def update_current_state(self):
        current_input_id_vecs = self.current_observation.get_sample_input_id_vecs()
        current_attn_mask_vecs = self.current_observation.get_sample_attn_mask_vecs()

        self.current_label = self.current_observation.get_label_enc()
        # self.current_state_input_id = self.current_input_id_vecs[self.current_state_ix]
        # self.current_state_attn_mask = self.current_attn_mask_vecs[self.current_state_ix]
        return {"input_id": current_input_id_vecs[self.current_state_ix].astype(int), 
                "attn_mask": current_attn_mask_vecs[self.current_state_ix].astype(int)}
    
    def _set_spaces(self):
        self.observation_space = spaces.Dict(
            {
                "input_id": spaces.Box(0, self.vocab_size, shape=(self.pool.window_size, ), dtype=int),
                "attn_mask": spaces.Box(0, 2, shape=(self.pool.window_size, ), dtype=int) 
            }
        )
    

    def step(self, action: int):
        reward, done, info = super().step(action)
        self.current_state = self.update_current_state()

        return self.current_state, reward, done, info 

    def reset(self):
        super().reset()
        self.current_state = self.update_current_state()
        return self.current_state
    
    def __len__(self) -> int:
        return len(self.pool)
  

class SimpleSequentialEnv(gym.Env):
    
    def __init__(self, 
                data_pool: SimpleSequentialDataPool,
                max_time_steps: int, 
                reward_fn: str = "f1",
                random_walk: bool = False):
        super().__init__()

        assert reward_fn in ["f1", "accuracy", "precision", "recall", "score"], \
            f"Reward functions needs to be one of 'f1', 'accuracy', 'precision', 'recall', 'score', got {reward_fn}."

        self.time_step = 0
        self.seen_samples = 0
        self.pool = data_pool
        self.random_walk = random_walk
        self.train_mode = True

        self.current_sample_ix = 0
        if self.random_walk:
            self.current_sample_ix = None
    
        self.current_observation, self.current_label = self.pool.create_episode(self.current_sample_ix)

        
        self.current_state_ix = 0
        self.current_state = self.current_observation[self.current_state_ix]

        action_list = copy.deepcopy(self.pool.possible_actions)
        action_list.append("<next>")
        # action_list.append("<previous>")
        self.action_space = ActionSpace(action_list)
        self.confusion_matrix = np.zeros((len(self.pool.possible_actions), len(self.pool.possible_actions)))
        self.last_reward = 0

        self.state_extractor = itv_models.DenseStateFeatureExtractor(self.pool.window_size, 3, [5, 5])
        self.next_predictor = itv_models.NextStatePredictor(3, [30, 30])
        self.action_predictor = itv_models.NextActionPredictor(3, [10, 10], len(self.action_space))
        self.state_loss = torch.nn.MSELoss()
        self.itv_optimizer = torch.optim.Adam(list(self.state_extractor.parameters()) + list(self.next_predictor.parameters()), lr = 0.001)
        
        
        self.max_time_steps = max_time_steps

        if reward_fn == "f1":
            self.reward_function = PartialReadingRewardF1(self.pool.possible_actions)
        elif reward_fn == "accuracy":
            self.reward_function = PartialReadingRewardAccuracy(self.pool.possible_actions)
        elif reward_fn == "precision":
            self.reward_function = PartialReadingRewardPrecision(self.pool.possible_actions)
        elif reward_fn == "recall":
            self.reward_function = PartialReadingRewardRecall(self.pool.possible_actions)
        else:
            self.reward_function = PartialReadingRewardScore(self.pool.possible_actions)
        self._set_spaces()


    def step(self, action: int):


        self.itv_optimizer.zero_grad()
        action_str = self.action_space.ix_to_action(action)
        label_str = self.current_label
        reward, self.confusion_matrix = self.reward_function(action_str, label_str, self.confusion_matrix, self.seen_samples+1)
        step_reward = reward * 0.5
        self.last_reward = reward

        if self.train_mode:
            state_now_phi = self.state_extractor(torch.Tensor(self.current_state))
            pred_next_phi = self.next_predictor(state_now_phi)
        
        if action_str in self.pool.possible_actions: # e.g action_str == "good" or "bad":
            if self.current_sample_ix != None:
                self.current_sample_ix += 1
                self.current_sample_ix %= len(self.pool)

            self.seen_samples += 1
            self.current_observation, self.current_label = self.pool.create_episode(self.current_sample_ix)
            self.current_state_ix = 0

        elif action_str == "<previous>":
            self.current_state_ix -= 1
        elif action_str == "<next>":
            self.current_state_ix += 1
            
        self.current_state_ix %= len(self.current_observation)
        self.current_state = self.current_observation[self.current_state_ix]

        state_next_phi = self.state_extractor(torch.Tensor(self.current_state))
        next_action_pred = self.action_predictor.predict(torch.cat([state_now_phi, state_next_phi], dim=-1))

        
        if self.train_mode:
            loss = self.state_loss(pred_next_phi, state_next_phi)
            loss.backward()
            self.itv_optimizer.step()
            step_reward += 0.5 * loss.item()



        self.time_step += 1
        done = (self.time_step > self.max_time_steps)
            
        
        dict_ = {"label": label_str, "action": action_str, "reward": step_reward}
        return self.current_state, step_reward, done, dict_ 

    def reset(self):
        if self.current_sample_ix != None:
            self.current_sample_ix = 0

        self.current_observation, self.current_label = self.pool.create_episode(self.current_sample_ix)
        
        self.current_state_ix = 0
        self.current_state = self.current_observation[self.current_state_ix]
        self.time_step = 0
        self.confusion_matrix = np.zeros((len(self.pool.possible_actions), len(self.pool.possible_actions)))
        self.last_reward = 0

        self.state_extractor = itv_models.DenseStateFeatureExtractor(self.pool.window_size, 10, [5, 5])
        self.next_predictor = itv_models.NextStatePredictor(self.pool.window_size, [30, 30])
        self.action_predictor = itv_models.NextActionPredictor(3, [10, 10], len(self.action_space))
        
        return self.current_state
    
    def set_train_mode(self, mode: bool):
        self.train_mode = mode
    
    def _set_spaces(self):
        low = np.full((self.pool.window_size, ), fill_value=-1)
        high = np.full((self.pool.window_size, ), fill_value=1)
        self.observation_space = spaces.Box(low, high, dtype=np.int32)
    
    def __len__(self) -> int:
        return len(self.pool)
     



if __name__ == "__main__":
    data = nlp_preprocessing.openDfFromPickle("NLP_datasets/ag_news/ag_news_train_distilbert-base-uncased.pkl")

    pool = PartialReadingDataPoolWithBertTokens(data, "text", "label", 512, 50, mask = True)
    env = TextEnvClfForBertModels(pool, 28996, int(1e+5), "score", True)
    check_env(env)
    print(env.current_observation)
    print(env.current_state)
    print("="*40)
    print(env.current_state)
    print("="*40)
    
    print("="*40)
    print(env.step(0))
    print(env.step(0))
    print(env.current_state)
    print("="*40)
    print(env.step(2))

    