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
# sys.path.append("/Users/emrebaloglu/Documents/RL/basic_reinforcement_learning") # macos
sys.path.append("C:\\Users\\mrbal\\Documents\\NLP\\RL\\basic_reinforcement_learning")


from NLP_utils import preprocessing as nlp_preprocessing
from NLP_utils import baseline_models as nlp_base_models
from NLP_utils import pytorch_datasets as nlp_datasets

import intrinsic_text_value_models as itv_models

from torch.utils.data import Dataset, DataLoader
from RL_for_NLP.text_data_pools import PartialReadingDataPoolWithTokens, PartialReadingDataPoolWithBertTokens, SimpleSequentialDataPool, PartialReadingDataPool
from RL_for_NLP.text_action_space import ActionSpace
from RL_for_NLP.text_reward_functions import PartialReadingRewardF1, PartialReadingRewardAccuracy, PartialReadingRewardPrecision, PartialReadingRewardRecall, PartialReadingRewardScore
from RL_for_NLP.observation import Observation

from abc import ABC, abstractmethod


class BaseTextEnvClf(gym.Env, ABC):
    def __init__(self, data_pool: PartialReadingDataPool, max_time_steps: int, reward_fn: str = "f1",  random_walk: bool = False):
        assert reward_fn in ["f1", "accuracy", "precision", "recall", "score"], \
            f"Reward functions needs to be one of 'f1', 'accuracy', 'precision', 'recall', 'score', got {reward_fn}."

        super().__init__()
        self.time_step = 0
        self.pool = data_pool
        self.random_walk = random_walk
        action_list = copy.deepcopy(self.pool.possible_actions)
        action_list.append("<next>")
        # action_list.append("<previous>")
        self.action_space = ActionSpace(action_list)
        self.confusion_matrix = np.zeros((len(self.pool.possible_actions), len(self.pool.possible_actions)))
        
        self.current_sample_ix = 0
        if self.random_walk:
            self.current_sample_ix = None
    
        self.current_observation = self.pool.create_episode(self.current_sample_ix)
        self.n_sentences_in_obs = len(self.current_observation.get_sample_vecs())

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
    # calculate the reward and update confusion matrix, given the action
    def calculate_reward(self, action: int) -> float: 
        action_str = self.action_space.ix_to_action(action)
        label_str = self.action_space.ix_to_action(self.current_observation.get_label_enc().item())
        reward, self.confusion_matrix = self.reward_function(action_str, label_str, self.confusion_matrix, 1)
        step_reward = reward
        if not isinstance(self.reward_function, PartialReadingRewardScore):
            step_reward = reward - self.last_reward
        self.last_reward = reward

        return step_reward
    
    # as the current state update varies on different envs, this should be overriden in subclass method
    # returns only reward, done, info (state index is also updated)
    def step(self, action: int) -> Tuple[float, bool, dict]:

        step_reward = self.calculate_reward(action)
        action_str = self.action_space.ix_to_action(action)
        label_str = self.action_space.ix_to_action(self.current_observation.get_label_enc().item())

        if action_str in self.pool.possible_actions: # action_str == "good" or "bad":
            if self.current_sample_ix != None:
                self.current_sample_ix += 1
                self.current_sample_ix %=  self.n_sentences_in_obs

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
    
    # here also, class params are reset but as the current state update varies on different envs, this should be overriden in subclass method
    def reset(self):
        if self.current_sample_ix != None:
            self.current_sample_ix = 0

        self.current_observation = self.pool.create_episode(self.current_sample_ix)
        self.current_state_ix = 0
        self.time_step = 0
        self.last_reward = 0
    

     

class TextEnvClf(BaseTextEnvClf):
    
    def __init__(self, data_pool: PartialReadingDataPoolWithTokens, max_time_steps: int, reward_fn: str = "f1", random_walk: bool = False):
        super().__init__(data_pool, max_time_steps, reward_fn, random_walk)
        self.current_state = self.update_current_state()
        self._set_spaces()
        
    # create/update the current state that the agent will observe, using current observation and state index
    def update_current_state(self) -> np.ndarray:
        current_vecs = self.current_observation.get_sample_vecs()
       
        return current_vecs[self.current_state_ix].astype(int)
    
    def _set_spaces(self):
       self.observation_space = spaces.Box(0, self.pool.vocab_size, shape=(self.pool.window_size, ), dtype=int) 
    
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

class TextEnvClfForBertModels(BaseTextEnvClf):
    
    def __init__(self, data_pool: PartialReadingDataPool, max_time_steps: int, reward_fn: str = "f1", random_walk: bool = False ):
        super().__init__()
        self.current_state = self.update_current_state()
        self._set_spaces()

    
    def update_current_state(self):
        current_input_id_vecs = self.current_observation.get_sample_input_id_vecs()
        current_attn_mask_vecs = self.current_observation.get_sample_attn_mask_vecs()

        self.current_label = self.current_observation.get_label_enc()
        self.current_state_ix = 0
        # self.current_state_input_id = self.current_input_id_vecs[self.current_state_ix]
        # self.current_state_attn_mask = self.current_attn_mask_vecs[self.current_state_ix]
        return {"input_id": current_input_id_vecs[self.current_state_ix], "attn_mask": current_attn_mask_vecs[self.current_state_ix]}
    
    def _set_spaces(self):
        self.observation_space = spaces.Dict(
            {
                "input_id": spaces.Box(0, self.pool.vocab_size, shape=(self.pool.window_size, ), dtype=int),
                "attn_mask": spaces.Box(0, 2, shape=(self.pool.window_size, ), dtype=int) 
            }
        )
    

    def step(self, action: int):
        """multiplier = self.same_sample_steps // self.pool.window_size
        if multiplier > 1:
            multiplier *= -1"""

        
        if action_str in self.pool.possible_actions: # e.g action_str == "good" or "bad":
            if self.current_sample_ix != None:
                self.current_sample_ix += 1
                self.current_sample_ix %= len(self.pool)

            self.seen_samples += 1
            self.current_observation = self.pool.create_episode(self.current_sample_ix)
            self.current_input_id_vecs = self.current_observation.get_sample_input_id_vecs()
            self.current_attn_mask_vecs = self.current_observation.get_sample_attn_mask_vecs()
           
            
            self.current_state_ix = 0

        elif action_str == "<previous>":
            self.current_state_ix -= 1
            self.same_sample_step += 1
        elif action_str == "<next>":
            self.current_state_ix += 1
            self.same_sample_step += 1
            
        self.current_state_ix %= len(self.current_input_id_vecs)
        self.current_state_input_id = self.current_input_id_vecs[self.current_state_ix]
        self.current_state_attn_mask = self.current_attn_mask_vecs[self.current_state_ix]
        
        

        if self.max_same_sample_steps < self.same_sample_step: # go to the next sample when limit is reached
            step_reward -= 5
            if self.current_sample_ix != None:
                self.current_sample_ix += 1
                self.current_sample_ix %= len(self.pool)

            self.seen_samples += 1
            self.current_observation = self.pool.create_episode(self.current_sample_ix)
            self.current_input_id_vecs = self.current_observation.get_sample_input_id_vecs()
            self.current_attn_mask_vecs = self.current_observation.get_sample_attn_mask_vecs()
           
            
            self.current_state_ix = 0
            self.same_sample_step = 0
        
        self.time_step += 1
        done = (self.time_step > self.max_time_steps)
            
        
        dict_ = {"text": self.current_observation.get_sample_str(), "label": label_str, "action": action_str, "reward": step_reward}
        if self.use_mask_tokens:
            return {"input_id": self.current_state_input_id.astype(np.int32), 
                    "attention_mask": self.current_state_attn_mask.astype(np.int32)}, \
                    step_reward, done, dict_ 
        return self.current_state_input_id.astype(np.int32), step_reward, done, dict_ 

    def reset(self):
        if self.current_sample_ix != None:
            self.current_sample_ix = 0

        self.current_observation = self.pool.create_episode(self.current_sample_ix)
        self.current_input_id_vecs = self.current_observation.get_sample_input_id_vecs()
        self.current_attn_mask_vecs = self.current_observation.get_sample_attn_mask_vecs()
      
        self.current_state_ix = 0
        self.current_state_input_id = self.current_input_id_vecs[self.current_state_ix]
        self.current_state_attn_mask = self.current_attn_mask_vecs[self.current_state_ix]
        self.time_step = 0
        self.same_sample_step = 0

        self.state_extractor = itv_models.RNN_Feature_Extractor(self.vocab_size, self.pool.window_size, 10)
        self.next_predictor = itv_models.NextStatePredictor(10, [30, 30])
        
        self.confusion_matrix = np.zeros((len(self.pool.possible_actions), len(self.pool.possible_actions)))
        self.last_reward = 0

        return self.current_state_input_id.astype(np.int32) # .detach().numpy()
    
    
    
    def __len__(self) -> int:
        return len(self.pool)
    
    def set_train_mode(self, mode: bool):
        self.train_mode = mode
    

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

        self.state_extractor = itv_models.DenseStateFeatureExtractor(self.pool.window_size, 10, [5, 5])
        self.next_predictor = itv_models.NextStatePredictor(self.pool.window_size, [30, 30])
        self.state_loss = torch.nn.MSELoss()
        self.itv_optimizer = torch.optim.Adam(list(self.state_extractor.parameters()) + list(self.next_predictor.parameters()), lr = 0.001)
        
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
        loss = self.state_loss(pred_next_phi, state_next_phi)
        if self.train_mode:
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
    data = nlp_preprocessing.openDfFromPickle("NLP_datasets/RT_Polarity/rt-polarity-train.pkl")
    pool = PartialReadingDataPoolWithTokens(data, "review", "label", 8)
    env = TextEnvClf(pool, int(1e+5), "score", True)
    check_env(env)
    print(env.current_observation)
    print(env.current_state)
    print(env.step(0))
    print(env.step(2))
    

    