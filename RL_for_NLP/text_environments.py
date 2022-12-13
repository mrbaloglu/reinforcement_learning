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

import intrinsic_text_value_models as itv_models

from torch.utils.data import Dataset, DataLoader
from RL_for_NLP.text_data_pools import PartialReadingDataPoolWithWord2Vec, PartialReadingDataPoolWithBertTokens, SimpleSequentialDataPool
from RL_for_NLP.text_action_space import ActionSpace
from RL_for_NLP.text_reward_functions import PartialReadingRewardF1, PartialReadingRewardAccuracy, PartialReadingRewardPrecision, PartialReadingRewardRecall, PartialReadingRewardScore
from RL_for_NLP.observation import Observation


class TextEnvClf(gym.Env):
    
    def __init__(self, 
                data_pool: PartialReadingDataPoolWithWord2Vec, 
                max_time_steps: int, 
                reward_fn: str = "f1",
                random_walk: bool = False):
        super().__init__()
        assert reward_fn in ["f1", "accuracy", "precision", "recall", "score"], \
            f"Reward functions needs to be one of 'f1', 'accuracy', 'precision', 'recall', 'score', got {reward_fn}."

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
        
        action_str = self.action_space.ix_to_action(action)
        label_str = self.action_space.ix_to_action(self.current_label.item())
        reward, self.confusion_matrix = self.reward_function(action_str, label_str, self.confusion_matrix)
        step_reward = reward
        if not isinstance(self.reward_function, PartialReadingRewardScore):
            step_reward = reward - self.last_reward
            
        self.last_reward = reward

        if action_str in self.pool.possible_actions: # action_str == "good" or "bad":
            if self.current_sample_ix != None:
                self.current_sample_ix += 1
                self.current_sample_ix %= len(self.pool)

            self.current_observation = self.pool.create_episode(self.current_sample_ix)
            self.current_vecs = self.current_observation.get_sample_vecs() 
            self.current_label = self.current_observation.get_label_enc()
            self.current_state_ix = 0

        elif action_str == "<previous>":
            self.current_state_ix -= 1
        elif action_str == "<next>":
            self.current_state_ix += 1
            
        self.current_state_ix %= len(self.current_vecs)
        self.current_state = self.current_vecs[self.current_state_ix]
        
        

        
        self.time_step += 1
        done = (self.time_step > self.max_time_steps)
            
        
        dict_ = {"text": self.current_observation.get_sample_str(), "label": label_str, "action": action_str, "reward": step_reward}

        return self.current_state.astype(np.int32), step_reward, done, dict_ 

    def reset(self):
        if self.current_sample_ix != None:
            self.current_sample_ix = 0

        self.current_observation = self.pool.create_episode(self.current_sample_ix)
        self.current_vecs = self.current_observation.get_sample_vecs()
        self.current_label = self.current_observation.get_label_enc()
        self.current_state_ix = 0
        self.current_state = self.current_vecs[self.current_state_ix]
        self.time_step = 0
        

        return self.current_state.astype(np.int32) # .detach().numpy()
    
    
    def _set_spaces(self):
        low = np.full((self.pool.window_size, ), fill_value=-1)
        high = np.full((self.pool.window_size, ), fill_value=int(1e+6))
        self.observation_space = spaces.Box(low, high, dtype=np.int32)

class TextEnvClfBert(gym.Env):
    
    def __init__(self, 
                data_pool: PartialReadingDataPoolWithBertTokens,
                max_time_steps: int,
                vocab_size: int,
                max_same_sample_steps: int = 15,
                reward_fn: str = "f1",
                random_walk: bool = False, 
                use_mask_tokens: bool = False):
        super().__init__()

        assert reward_fn in ["f1", "accuracy", "precision", "recall", "score"], \
            f"Reward functions needs to be one of 'f1', 'accuracy', 'precision', 'recall', 'score', got {reward_fn}."

        self.time_step = 0
        self.seen_samples = 0
        self.same_sample_step = 0
        self.max_same_sample_steps = max_same_sample_steps
        self.pool = data_pool
        self.random_walk = random_walk
        self.use_mask_tokens = use_mask_tokens
        self.vocab_size = vocab_size

        self.state_extractor = itv_models.RNN_Feature_Extractor(self.vocab_size, self.pool.window_size, 10)
        self.next_predictor = itv_models.NextStatePredictor(10, [30, 30])
        self.state_loss = torch.nn.MSELoss()
        self.itv_optimizer = torch.optim.Adam(list(self.state_extractor.parameters()) + list(self.next_predictor.parameters()), lr = 0.001)
        
        
        self.train_mode = True
        
        self.current_sample_ix = 0
        if self.random_walk:
            self.current_sample_ix = None
    
        self.current_observation = self.pool.create_episode(self.current_sample_ix)

        self.current_input_id_vecs = self.current_observation.get_sample_input_id_vecs()
        self.current_attn_mask_vecs = self.current_observation.get_sample_attn_mask_vecs()

        self.current_label = self.current_observation.get_label_enc()
        self.current_state_ix = 0
        self.current_state_input_id = self.current_input_id_vecs[self.current_state_ix]
        self.current_state_attn_mask = self.current_attn_mask_vecs[self.current_state_ix]
        
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
        """multiplier = self.same_sample_steps // self.pool.window_size
        if multiplier > 1:
            multiplier *= -1"""

        action_str = self.action_space.ix_to_action(action)
        label_str = self.action_space.ix_to_action(self.current_label.item())
        reward, self.confusion_matrix = self.reward_function(action_str, label_str, self.confusion_matrix, 1)
        step_reward = reward # * 0.5
        self.last_reward = reward
        
        if self.train_mode:
            self.itv_optimizer.zero_grad()
            state_now = torch.from_numpy(self.current_state_input_id.copy())
            state_now_phi = self.state_extractor(state_now)
            pred_next_phi = self.next_predictor(state_now_phi)
        
        if action_str in self.pool.possible_actions: # e.g action_str == "good" or "bad":
            if self.current_sample_ix != None:
                self.current_sample_ix += 1
                self.current_sample_ix %= len(self.pool)

            self.seen_samples += 1
            self.current_observation = self.pool.create_episode(self.current_sample_ix)
            self.current_input_id_vecs = self.current_observation.get_sample_input_id_vecs()
            self.current_attn_mask_vecs = self.current_observation.get_sample_attn_mask_vecs()
            self.current_label = self.current_observation.get_label_enc()
            
            self.current_state_ix = 0
            self.same_sample_step = 0

        elif action_str == "<previous>":
            self.current_state_ix -= 1
            self.same_sample_step += 1
        elif action_str == "<next>":
            self.current_state_ix += 1
            self.same_sample_step += 1
            
        self.current_state_ix %= len(self.current_input_id_vecs)
        self.current_state_input_id = self.current_input_id_vecs[self.current_state_ix]
        self.current_state_attn_mask = self.current_attn_mask_vecs[self.current_state_ix]
        
        if self.train_mode:
            state_next = torch.from_numpy(self.current_state_input_id.copy())
            state_next_phi = self.state_extractor(state_next)
            loss = self.state_loss(pred_next_phi, state_next_phi)
            loss.backward()
            self.itv_optimizer.step()
            step_reward += 0.5 * loss.item()
        

        if self.max_same_sample_steps < self.same_sample_step: # go to the next sample when limit is reached
            step_reward -= 5
            if self.current_sample_ix != None:
                self.current_sample_ix += 1
                self.current_sample_ix %= len(self.pool)

            self.seen_samples += 1
            self.current_observation = self.pool.create_episode(self.current_sample_ix)
            self.current_input_id_vecs = self.current_observation.get_sample_input_id_vecs()
            self.current_attn_mask_vecs = self.current_observation.get_sample_attn_mask_vecs()
            self.current_label = self.current_observation.get_label_enc()
            
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
        self.current_label = self.current_observation.get_label_enc()
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
    
    def _set_spaces(self):
        low = np.full((self.pool.window_size, ), fill_value=-1)
        high = np.full((self.pool.window_size, ), fill_value=int(1e+6))
        self.observation_space = spaces.Box(low, high, dtype=np.int32)
    
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
    # data = nlp_preprocessing.openDfFromPickle("NLP_datasets/RT_Polarity/rt-polarity-train-bert.pkl")
    # pool = PartialReadingDataPoolWithBertTokens(data, "review", "label", "good", 8)
    # env = TextEnvClfBert(pool, int(1e+5), "precision", True, True)
    pool = SimpleSequentialDataPool(10000, 10, 5)
    env = SimpleSequentialEnv(pool, 500)
    check_env(env)
    print("==== Start =====")
    print(env.current_observation, env.current_label)
    print(env.step(1))
    print("==== step 1 taken =====")
    print(env.current_observation, env.current_label)
    print(env.step(2))
    print("==== step 2 taken =====")
    print(env.current_observation, env.current_label)
    print(env.step(1))
    print("==== step 1 taken =====")
    print(env.current_observation, env.current_label)
    print(env.current_state.shape)

    