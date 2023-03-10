import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

import numpy as np

from abc import ABC, abstractmethod
from typing import Union, Dict, Tuple, Any, Optional
from collections import namedtuple
from itertools import count
import gym
from tqdm import tqdm
from transformers import DistilBertModel
from collections import Counter
import gc
 

from RL_for_NLP.text_reward_functions import calculate_stats_from_cm

SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])

#### Policy implementations with torch neural networks ##########################
################################################################################
################################################################################
class ActorCriticPolicy(nn.Module, ABC):
    def __init__(self, state_dim: int, action_dim: int, dropout: float = 0.4) -> None:
        super().__init__()
        self.dropout = dropout
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.saved_log_probs = []
        self.rewards = []
        self.saved_actions = []

    # feed-forward neural network definition
    @abstractmethod
    def forward(self, x: Union[th.Tensor, np.ndarray]) -> th.Tensor:
        raise NotImplementedError
    


class DenseActorCriticPolicy(ActorCriticPolicy):
    def __init__(self, state_dim: int, action_dim: int, dropout = 0.1):
        super().__init__(state_dim=state_dim, action_dim=action_dim, dropout=dropout)
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.affine1 = nn.Linear(self.state_dim, 256)
        self.affine2 = nn.Linear(256, 128)
        self.affine3 = nn.Linear(128, 64)
        self.dropout_layer = nn.Dropout(p=self.dropout)
        # actor's layer
        self.action_head = nn.Linear(64, self.action_dim)

        # critic's layer
        self.value_head = nn.Linear(64, 1)

        

    def forward(self, x: Union[th.Tensor, np.ndarray]) -> th.Tensor:
        if not isinstance(x, th.Tensor):
            x = th.from_numpy(x)
        
        x = self.affine1(x)
        x = F.relu(x)
        x = self.dropout_layer(x)
        x = F.relu(self.affine2(x))
        x = F.relu(self.affine3(x))
        # actor: choses action to take from state s_t
        # by returning probability of each action
        action_prob = F.softmax(self.action_head(x), dim=-1)

        # critic: evaluates being in the state s_t
        state_values = self.value_head(x)

        # return values for both actor and critic as a tuple of 2 values:
        # 1. a list with the probability of each action over the action space
        # 2. the value from state s_t
        return action_prob, state_values

class DistibertActorCriticPolicy(ActorCriticPolicy):
    def __init__(self, state_dim: int, action_dim: int, use_last_n_layers: int = 0, dropout = 0.4, bert_model = None):
        super().__init__(state_dim=state_dim, action_dim=action_dim, dropout=dropout)
        
        if bert_model == None:
            self.bert = DistilBertModel.from_pretrained("distilbert-base-uncased")
        else: 
            self.bert = bert_model
        """n_distilbert_param = len([x for x in self.distilbert.parameters()])
        cnt = 0
        for param in self.distilbert.parameters():
            if cnt < use_last_n_layers:
                param.requires_grad = False
            cnt += 1"""
        self.bert.requires_grad = False
        
        
        self.flatten = nn.Flatten()
        self.dense = nn.Linear(self.state_dim*768, 128)
        self.dropout_layer = nn.Dropout(p=self.dropout)
        # actor's layer
        self.action_head = nn.Linear(128, self.action_dim)

        # critic's layer
        self.value_head = nn.Linear(128, 1)

        

    def forward(self, x_input_id: th.tensor, x_attn_mask: th.tensor) -> th.Tensor:
        
        x = self.bert(x_input_id, x_attn_mask).last_hidden_state
        
        x = self.flatten(x)
        x = F.relu(self.dense(x))
        x = self.dropout_layer(x)
        # actor: choses action to take from state s_t
        # by returning probability of each action
        action_prob = F.softmax(self.action_head(x), dim=-1)

        # critic: evaluates being in the state s_t
        state_values = self.value_head(x)

        # return values for both actor and critic as a tuple of 2 values:
        # 1. a list with the probability of each action over the action space
        # 2. the value from state s_t
        return action_prob, state_values
    
    """def _apply(self, fn):
        self.bert = fn(self.bert)
        self.flatten = fn(self.flatten)
        self.dense = fn(self.dense)
        self.dropout_layer = fn(self.dropout_layer)
        self.action_head = fn(self.action_head)
        self.value_head = fn(self.value_head)

        return self"""

class BertFeatureExtractor(nn.Module):
    def __init__(self, max_len: int = 512, bert_model = None, freeze = True):
        super().__init__()
        
        if bert_model == None:
            self.bert = DistilBertModel.from_pretrained("distilbert-base-uncased")
        else: 
            self.bert = bert_model
        
        if freeze:
            self.bert.requires_grad = True
        
        self.flatten = nn.Flatten()
        self.out_dim = 0

        dummy_input_id = th.randint(1, 4, (3, max_len)).long()
        dummy_mask = th.ones_like(dummy_input_id).long()
        with th.no_grad():
            dummy_out = self.bert(dummy_input_id, dummy_mask).last_hidden_state
            dummy_out = self.flatten(dummy_out)
        
        self.out_dim = dummy_out.shape[-1]

        del dummy_input_id, dummy_mask, dummy_out
        
        

    def forward(self, x_input_id: th.tensor, x_attn_mask: th.tensor) -> th.Tensor:
        
        x = self.bert(x_input_id, x_attn_mask).last_hidden_state
        x = self.flatten(x)

        return x
    
    """def _apply(self, fn):
        self.bert = fn(self.bert)
        self.flatten = fn(self.flatten)
        self.dense = fn(self.dense)
        self.dropout_layer = fn(self.dropout_layer)
        self.action_head = fn(self.action_head)
        self.value_head = fn(self.value_head)

        return self"""
 

################################################################################
################################################################################
################################################################################


#################### algorithm implementations #################################
################################################################################
################################################################################
class ActorCriticAlgorithm:
    def __init__(self, policy: ActorCriticPolicy, env: gym.Env, optimizer: th.optim, device: th.device = th.device("cpu"),
            gamma: float = 0.99, **kwargs):
        self.policy = policy
        self.env = env
        self.gamma = gamma
        self.optimizer = optimizer
        self.device = device
        self.policy.to(self.device)
    

    # select an action given a state from env, using the policy
    def select_action(self, state: np.ndarray):
        state = th.from_numpy(state).float().unsqueeze(0).to(self.device)
        probs, state_value = self.policy(state)
        m = Categorical(probs)
        action = m.sample()
        # save to action buffer
        self.policy.saved_actions.append(SavedAction(m.log_prob(action), state_value))
        self.policy.saved_log_probs.append(m.log_prob(action))
        return action.item()

     
 
    def finish_episode(self, epsilon: float = np.finfo(np.float32).eps.item()): # epsilon > 0 to prevent divide by 0 
        """
        Training code. Calculates actor and critic loss and performs backprop.
        """
        R = 0
        saved_actions = self.policy.saved_actions
        policy_losses = [] # list to save actor (policy) loss
        value_losses = [] # list to save critic (value) loss
        returns = [] # list to save the true values

        # calculate the true value using rewards returned from the environment
        for r in self.policy.rewards[::-1]:
            # calculate the discounted value
            R = r + self.gamma * R
            returns.insert(0, R)

        returns = th.tensor(returns).to(self.device)
        returns = (returns - returns.mean()) / (returns.std() + epsilon)

        for (log_prob, value), R in zip(saved_actions, returns):
            advantage = R - value.item()

            # calculate actor (policy) loss
            policy_losses.append(-log_prob * advantage)

            # calculate critic (value) loss using L1 smooth loss
            value_losses.append(F.smooth_l1_loss(value, th.tensor([R]).to(self.device)))

        # reset gradients
        self.optimizer.zero_grad()

        # sum up all the values of policy_losses and value_losses
        loss = th.stack(policy_losses).sum() + th.stack(value_losses).to(self.device).sum()

        # perform backprop
        loss.backward()
        self.optimizer.step()

        # reset rewards and action buffer
        del self.policy.rewards[:]
        del self.policy.saved_actions[:]

    def train_a2c(self, n_episodes: int, n_steps: int, render_env: bool = False, log_interval: int = 150):
        running_reward = 10
        last_reward = 0
        # run infinitely many episodes / number of episodes selected
        pbar = tqdm(range(1, n_episodes+1))
        for i_episode in pbar:
            pbar.set_description(f"Average reward so far: {last_reward:.3f} (updated every {log_interval} episodes)")
            # reset environment and episode reward
            state = self.env.reset()
            ep_reward = 0

            # for each episode, only run 9999 steps so that we don't
            # infinite loop while learning
            for t in tqdm(range(1, n_steps), leave=False):

                # select action from policy
                action = self.select_action(state)

                # take the action
                state, reward, done, _, = self.env.step(action)

                if render_env:
                    self.env.render()

                self.policy.rewards.append(reward)
                ep_reward += reward
                if done:
                    break

            # update cumulative reward
            running_reward = 0.05 * ep_reward + (1 - 0.05) * running_reward

            
            # perform backprop
            self.finish_episode()

            # log results
            if i_episode % log_interval == 0:
                # print(f'Episode {i_episode}\tLast reward: {ep_reward:.2f}\tAverage reward: {running_reward:.2f}')
                last_reward = running_reward

            """# check if we have "solved" the cart pole problem
            if running_reward > self.env.spec.reward_threshold:
                print("Solved! Running reward is now {} and "
                    "the last episode runs to {} time steps!".format(running_reward, t))
                break"""
        

class ActorCriticAlgorithmBertModel:
    def __init__(self, policy: ActorCriticPolicy, env: gym.Env, optimizer: th.optim, device: th.device = th.device("cpu"),
             gamma: float = 0.99, **kwargs):
        self.policy = policy
        self.env = env
        self.gamma = gamma
        self.optimizer = optimizer
        self.device = device
    
        self.policy.to(self.device)

    # select an action given a state from env, using the policy
    def select_action(self, state: Dict[str, np.ndarray]):
        input_ids = th.from_numpy(state["input_id"]).unsqueeze(0).to(self.device)
        attn_mask = th.from_numpy(state["attn_mask"]).unsqueeze(0).to(self.device)
        
        probs, state_value = self.policy(input_ids, attn_mask)
        m = Categorical(probs)
        action = m.sample()

        # save to action buffer
        self.policy.saved_actions.append(SavedAction(m.log_prob(action), state_value))
        self.policy.saved_log_probs.append(m.log_prob(action))
        action_ = action.cpu().item()

        del input_ids, attn_mask, probs, state_value, m, action
        th.cuda.empty_cache()
        return action_

     
 
    def finish_episode(self, epsilon: float = np.finfo(np.float32).eps.item()): # epsilon > 0 to prevent divide by 0 
        """
        Training code. Calculates actor and critic loss and performs backprop.
        """

        R = 0
        saved_actions = self.policy.saved_actions
        policy_losses = [] # list to save actor (policy) loss
        value_losses = [] # list to save critic (value) loss
        returns = [] # list to save the true values

        # calculate the true value using rewards returned from the environment
        for r in self.policy.rewards[::-1]:
            # calculate the discounted value
            R = r + self.gamma * R
            returns.insert(0, R)

        returns = th.tensor(returns).to(self.device)
        returns = (returns - returns.mean()) / (returns.std() + epsilon)

        for (log_prob, value), R in zip(saved_actions, returns):
            advantage = R - value.item()

            # calculate actor (policy) loss
            policy_losses.append(-log_prob * advantage)

            # calculate critic (value) loss using L1 smooth loss
            value_losses.append(F.smooth_l1_loss(value, th.tensor([R]).to(self.device)))

        # reset gradients
        self.optimizer.zero_grad()

        # sum up all the values of policy_losses and value_losses
        loss = th.stack(policy_losses).sum() + th.stack(value_losses).to(self.device).sum()

        # perform backprop
        loss.backward()
        self.optimizer.step()

        # reset rewards and action buffer
        del self.policy.rewards[:]
        del self.policy.saved_actions[:]

        del loss, policy_losses, value_losses
        th.cuda.empty_cache()
        
    def train_a2c(self, n_episodes: int, n_steps: int, render_env: bool = False, log_interval: int = 150):
        self.policy.to(self.device)
        running_reward = 0.
        last_reward = 0
        # run infinitely many episodes / number of episodes selected
        pbar = tqdm(range(1, n_episodes+1))
        for i_episode in pbar:
            stats = calculate_stats_from_cm(self.env.confusion_matrix)
            pbar.set_description(f"Accuracy: {stats['accuracy']:.2f}, Precision: {stats['precision']:.2f}, Recall: {stats['recall']:.2f}, F1: {stats['f1']:.2f}, Average reward: {last_reward:.3f} (updated every {log_interval} episodes)")
            # reset environment and episode reward
            state = self.env.reset()
            ep_reward = 0

            # for each episode, only run 9999 steps so that we don't
            # infinite loop while learning
            for t in range(1, n_steps):

                # select action from policy
                action = self.select_action(state)

                # take the action
                state, reward, done, _, = self.env.step(action)

                if render_env:
                    self.env.render()

                self.policy.rewards.append(reward)
                ep_reward += reward
                if done:
                    break

            # update cumulative reward
            running_reward = 0.05 * ep_reward + (1 - 0.05) * running_reward

            
            # perform backprop
            self.finish_episode()
            th.cuda.empty_cache()
            # log results
            if i_episode % log_interval == 0:
                # print(f'Episode {i_episode}\tLast reward: {ep_reward:.2f}\tAverage reward: {running_reward:.2f}')
                last_reward = ep_reward
            
            # print(f"Observe: {_}")

            """# check if we have "solved" the cart pole problem
            if running_reward > self.env.spec.reward_threshold:
                print("Solved! Running reward is now {} and "
                    "the last episode runs to {} time steps!".format(running_reward, t))
                break"""
    
    def eval_model(self, env, total_timesteps=100):
        done = False
        obs = env.reset()
        total_reward = 0.0
        actions = []
        seen_samples = 0
        self.policy.to(self.device)
        for _ in tqdm(range(total_timesteps)):
            action = self.select_action(obs)
            obs, rewards, done, info = env.step(action)
            action = env.action_space.ix_to_action(action)
            if action in env.pool.possible_actions:
                seen_samples += 1
            if done:
                obs = env.reset()
        
            del self.policy.rewards[:]
            del self.policy.saved_actions[:]
            actions.append(action)
            total_reward += rewards
            del rewards, done, info, action
      
        print("---------------------------------------------");
        print(f"Total Steps and seen samples: {len(actions), seen_samples}")
        print(f"Total reward: {total_reward}")
        print(f"Stats:  {calculate_stats_from_cm(env.confusion_matrix)}")
        acts = list(Counter(actions).keys())
        freqs = list(Counter(actions).values())
        total = len(actions)
        print(f"Action stats --  {[{acts[ii]: freqs[ii]/total} for ii in range(len(acts))]}")
        print("---------------------------------------------")


class ActorCriticAlgorithmControlBertModel:
    def __init__(self, stop_policy: ActorCriticPolicy, clf_policy: ActorCriticPolicy, next_policy: ActorCriticPolicy, 
                 feature_extractor: nn.Module, env: gym.Env, stop_optimizer: th.optim,  clf_optimizer: th.optim,  
                 next_optimizer: th.optim, device: th.device = th.device("cpu"), gamma: float = 0.99, **kwargs):
        self.clf_policy = clf_policy # policy that selects the classifier actions
        self.stop_policy = stop_policy # policy that selects whether to stop for classification or continue reading
        self.next_policy = next_policy # policy that selects the skip size when reading (skip 0: reread, skip 1: read next, skip 2: read 2 next, ...)
        self.env = env
        self.gamma = gamma
        # self.stop_optimizer = stop_optimizer -> optimize stop_policy parameters in clf and next optimizers
        self.clf_optimizer = clf_optimizer
        self.next_optimizer = next_optimizer
        self.feature_extractor = feature_extractor

        self.device = device
    
        self.stop_policy.to(self.device)
        self.clf_policy.to(self.device)
        self.next_policy.to(self.device)

    # select an action given a state from env, using the policy
    def select_action(self, state: Dict[str, np.ndarray], info: Optional[Dict[str, Any]]) -> Tuple[int, str]:
        
        input_ids = th.from_numpy(state["input_id"]).unsqueeze(0).to(self.device)
        attn_mask = th.from_numpy(state["attn_mask"]).unsqueeze(0).to(self.device)
        extracted_features = self.feature_extractor(input_ids, attn_mask)
        last_action = None
        action_ = None
        action_type = None
        if info:
            last_action = info["action"]
            if last_action == "<stop>":
                # clf policy actions
                probs, state_value = self.clf_policy(extracted_features)
                m = Categorical(probs)
                action = m.sample()

                # save to action buffer
                self.clf_policy.saved_actions.append(SavedAction(m.log_prob(action), state_value))
                self.clf_policy.saved_log_probs.append(m.log_prob(action))
                action_ = action.cpu().item()
                action_type = "clf"
                
            elif last_action == "<continue>":
                # next policy actions (decide to reread or how much to skip)
                probs, state_value = self.next_policy(extracted_features)
                m = Categorical(probs)
                action = m.sample()

                # save to action buffer
                self.next_policy.saved_actions.append(SavedAction(m.log_prob(action), state_value))
                self.next_policy.saved_log_probs.append(m.log_prob(action))
                action_ = action.cpu().item()
                action_type = "next"

            else:
                # decide to continue reading or stop
                probs, state_value = self.stop_policy(extracted_features)
                m = Categorical(probs)
                action = m.sample()
                # save to action buffer
                self.stop_policy.saved_actions.append(SavedAction(m.log_prob(action), state_value))
                self.stop_policy.saved_log_probs.append(m.log_prob(action))
                action_ = action.cpu().item()
                action_type = "c/s"
        else:
            # decide to continue reading or stop
            probs, state_value = self.stop_policy(extracted_features)
            m = Categorical(probs)
            action = m.sample()
            # save to action buffer
            self.stop_policy.saved_actions.append(SavedAction(m.log_prob(action), state_value))
            self.stop_policy.saved_log_probs.append(m.log_prob(action))
            action_ = action.cpu().item()
            action_type = "c/s"

        return action_, action_type

     
 
    def finish_episode(self, epsilon: float = np.finfo(np.float32).eps.item()): # epsilon > 0 to prevent divide by 0 
        """
        Training code. Calculates actor and critic loss and performs backprop.
        """
        for policy, optimizer in zip([self.clf_policy, self.next_policy], 
                                     [self.clf_optimizer, self.next_optimizer]): 
            R = 0
            saved_actions = policy.saved_actions
            policy_losses = [] # list to save actor (policy) loss
            value_losses = [] # list to save critic (value) loss
            returns = [] # list to save the true values

            # calculate the true value using rewards returned from the environment
            for r in policy.rewards[::-1]:
                # calculate the discounted value
                R = r + self.gamma * R
                returns.insert(0, R)

            returns = th.tensor(returns).to(self.device)
            returns = (returns - returns.mean()) / (returns.std() + epsilon)

            for (log_prob, value), R in zip(saved_actions, returns):
                advantage = R - value.item()

                # calculate actor (policy) loss
                policy_losses.append(-log_prob * advantage)

                # calculate critic (value) loss using L1 smooth loss
                value_losses.append(F.smooth_l1_loss(value, th.tensor([R]).to(self.device)))

            # reset gradients
            optimizer.zero_grad()

            # sum up all the values of policy_losses and value_losses
            loss = th.stack(policy_losses).sum() + th.stack(value_losses).to(self.device).sum()

            # perform backprop
            loss.backward()
            optimizer.step()

            # reset rewards and action buffer
            del policy.rewards[:]
            del policy.saved_actions[:]

            del loss, policy_losses, value_losses
            th.cuda.empty_cache()
    
    def action_ix2str(self, action, action_type) -> str:
        action_str = None

        if action_type == "c/s":
            action_str = self.env.action_space.ix_to_action(action)
        elif action_type == "next":
            action_str = self.env.n_action_space.ix_to_action(action)
        elif action_type == "clf":
            action_str = self.env.clf_action_space.ix_to_action(action)
        else:
            raise ValueError(f"Invalid action type encountered: action_type -> {action_type}")
        
        return action_str

    def train_a2c(self, n_episodes: int, n_steps: int, render_env: bool = False, log_interval: int = 150):
        self.stop_policy.to(self.device)
        self.next_policy.to(self.device)
        self.clf_policy.to(self.device)
        state = self.env.reset()
        running_reward = 0.
        last_reward = 0
        # run infinitely many episodes / number of episodes selected
        pbar = tqdm(range(1, n_episodes+1))
        for i_episode in pbar:
            stats = calculate_stats_from_cm(self.env.confusion_matrix)
            pbar.set_description(f"Accuracy: {stats['accuracy']:.2f}, Precision: {stats['precision']:.2f}, Recall: {stats['recall']:.2f}, F1: {stats['f1']:.2f}, Average reward: {last_reward:.3f} (updated every {log_interval} episodes)")
            # reset environment and episode reward
            
            ep_reward = 0
            info = None
            # for each episode, only run 9999 steps so that we don't
            # infinite loop while learning
            for t in range(1, n_steps):

                # select action from policy
                action, action_type = self.select_action(state, info)
                action_str = None
                action_str = self.action_ix2str(action, action_type)
                # take the action
                state, reward, done, info = self.env.step(action_str)

                if render_env:
                    self.env.render()

                if info["action"] in self.env.clf_action_space.actions:
                    self.clf_policy.rewards.append(reward)
                elif info["action"] in self.env.n_action_space.actions:
                    self.next_policy.rewards.append(reward)

                
                self.stop_policy.rewards.append(reward)
                ep_reward += reward
                if done:
                    break

            # update cumulative reward
            running_reward += ep_reward

            
            # perform backprop
            self.finish_episode()
            th.cuda.empty_cache()
            # log results
            if i_episode % log_interval == 0:
                last_reward = running_reward / log_interval
                running_reward = 0.
            
            # print(f"Observe: {_}")

            """# check if we have "solved" the cart pole problem
            if running_reward > self.env.spec.reward_threshold:
                print("Solved! Running reward is now {} and "
                    "the last episode runs to {} time steps!".format(running_reward, t))
                break"""
    
    def eval_model(self, env, total_timesteps=100):
        done = False
        obs = env.reset()
        total_reward = 0.0
        actions = []
        seen_samples = 0
        self.stop_policy.to(self.device)
        self.clf_policy.to(self.device)
        self.next_policy.to(self.device)
        info = None
        for _ in tqdm(range(total_timesteps)):
            action, action_type = self.select_action(obs, info)
            action_str = self.action_ix2str(action, action_type)
            obs, rewards, done, info = env.step(action_str)
            
            if action in env.clf_action_space.actions:
                seen_samples += 1
            if done:
                obs = env.reset()
        
            del self.stop_policy.rewards[:], self.clf_policy.rewards[:], self.next_policy.rewards[:]
            del self.stop_policy.saved_actions[:], self.clf_policy.saved_actions[:], self.next_policy.saved_actions[:]
            actions.append(action_str)
            total_reward += rewards
            del rewards, done, action
      
        print("---------------------------------------------");
        print(f"Total Steps and seen samples: {len(actions), seen_samples}")
        print(f"Total reward: {total_reward}")
        print(f"Stats:  {calculate_stats_from_cm(env.confusion_matrix)}")
        acts = list(Counter(actions).keys())
        freqs = list(Counter(actions).values())
        total = len(actions)
        print(f"Action stats --  {[{acts[ii]: freqs[ii]/total} for ii in range(len(acts))]}")
        print("---------------------------------------------")


if __name__ == "__main__":

    env = gym.make('MountainCar-v0')
    print(env.observation_space, env.observation_space.shape)
    print(env.action_space, env.action_space.shape)
    print(env.reset())
    th.manual_seed(42)

    model = DenseActorCriticPolicy(env.observation_space.shape[0], env.action_space.n)
    optimizer = th.optim.Adam(model.parameters(), lr=3e-2)
    
    algo = ActorCriticAlgorithm(model, env, optimizer, device=th.device("cuda"))
    
    for _ in range(5):
        algo.train_a2c(10000, 50, log_interval=10)