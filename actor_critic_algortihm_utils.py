import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

import numpy as np

from abc import ABC, abstractmethod
from typing import Union, Dict, Tuple
from collections import namedtuple
from itertools import count
import gym
from tqdm import tqdm
from transformers import DistilBertModel

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
    def __init__(self, state_dim: int, action_dim: int, dropout = 0.4):
        super().__init__(state_dim=state_dim, action_dim=action_dim, dropout=dropout)
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.affine1 = nn.Linear(self.state_dim, 128)
        self.affine2 = nn.Linear(128, 15000)
        self.affine3 = nn.Linear(15000, 128)
        self.dropout_layer = nn.Dropout(p=self.dropout)
        # actor's layer
        self.action_head = nn.Linear(128, self.action_dim)

        # critic's layer
        self.value_head = nn.Linear(128, 1)

        

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
        self.distilbert.requires_grad = False
        
        
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

################################################################################
################################################################################
################################################################################


#################### algorithm implementations #################################
################################################################################
################################################################################
class ActorCriticAlgorithm:
    def __init__(self, policy: ActorCriticPolicy, env: gym.Env, optimizer: th.optim, gamma: float = 0.99, **kwargs):
        self.policy = policy
        self.env = env
        self.gamma = gamma
        self.optimizer = optimizer
    
    

    # select an action given a state from env, using the policy
    def select_action(self, state: np.ndarray):
        state = th.from_numpy(state).float().unsqueeze(0)
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

        returns = th.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + epsilon)

        for (log_prob, value), R in zip(saved_actions, returns):
            advantage = R - value.item()

            # calculate actor (policy) loss
            policy_losses.append(-log_prob * advantage)

            # calculate critic (value) loss using L1 smooth loss
            value_losses.append(F.smooth_l1_loss(value, th.tensor([R])))

        # reset gradients
        self.optimizer.zero_grad()

        # sum up all the values of policy_losses and value_losses
        loss = th.stack(policy_losses).sum() + th.stack(value_losses).sum()

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
    def __init__(self, policy: ActorCriticPolicy, env: gym.Env, optimizer: th.optim, gamma: float = 0.99, **kwargs):
        self.policy = policy
        self.env = env
        self.gamma = gamma
        self.optimizer = optimizer
    
    

    # select an action given a state from env, using the policy
    def select_action(self, state: Dict[str, np.ndarray]):
        input_ids = th.from_numpy(state["input_id"]).unsqueeze(0)
        attn_mask = th.from_numpy(state["attn_mask"]).unsqueeze(0)
        
        probs, state_value = self.policy(input_ids, attn_mask)
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

        returns = th.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + epsilon)

        for (log_prob, value), R in zip(saved_actions, returns):
            advantage = R - value.item()

            # calculate actor (policy) loss
            policy_losses.append(-log_prob * advantage)

            # calculate critic (value) loss using L1 smooth loss
            value_losses.append(F.smooth_l1_loss(value, th.tensor([R])))

        # reset gradients
        self.optimizer.zero_grad()

        # sum up all the values of policy_losses and value_losses
        loss = th.stack(policy_losses).sum() + th.stack(value_losses).sum()

        # perform backprop
        loss.backward()
        self.optimizer.step()

        # reset rewards and action buffer
        del self.policy.rewards[:]
        del self.policy.saved_actions[:]

    def train_a2c(self, n_episodes: int, n_steps: int, render_env: bool = False, log_interval: int = 150):
        running_reward = 0.
        last_reward = 0
        # run infinitely many episodes / number of episodes selected
        pbar = tqdm(range(1, n_episodes+1))
        for i_episode in pbar:
            pbar.set_description(f"Stats:  {calculate_stats_from_cm(self.env.confusion_matrix)}, Average reward: {last_reward:.3f} (updated every {log_interval} episodes)")
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
                last_reward = ep_reward
            
            print(f"Observe: {_}")

            """# check if we have "solved" the cart pole problem
            if running_reward > self.env.spec.reward_threshold:
                print("Solved! Running reward is now {} and "
                    "the last episode runs to {} time steps!".format(running_reward, t))
                break"""
        

if __name__ == "__main__":

    env = gym.make('MountainCar-v0')
    print(env.observation_space, env.observation_space.shape)
    print(env.action_space, env.action_space.shape)
    print(env.reset())
    th.manual_seed(42)

    model = DenseActorCriticPolicy(env.observation_space.shape[0], env.action_space.n)
    optimizer = th.optim.Adam(model.parameters(), lr=3e-2)
    
    algo = ActorCriticAlgorithm(model, env, optimizer)
    algo.train_a2c(10000, 1000, log_interval=10)