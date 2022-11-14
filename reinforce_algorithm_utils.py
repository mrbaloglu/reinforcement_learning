from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from tqdm import tqdm

import numpy as np
from collections import deque

from typing import List, Union
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score


class Softmax_Policy_Dense_Layers(nn.Module):

    def __init__(self, state_size: int, action_size: int, hidden_layer_dims: List[int]):
        """
        Initialize a policy with softmax probabilities, estimated with a fully-connected neural network.

        Args:
            state_size (int): Size of the states.
            action_size (int): Size of the actions.
            hidden_layer_dims (List[int]): List of units in hidden layers. 
        """
        super(Softmax_Policy_Dense_Layers, self).__init__()

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}.")
        self.hidden_layers = []
        if len(hidden_layer_dims) > 0:
            self.fc1 = nn.Linear(state_size, hidden_layer_dims[0]).to(self.device)
            for ii in range(len(hidden_layer_dims) - 1):
                hidden_layer = nn.Linear(hidden_layer_dims[ii], hidden_layer_dims[ii+1]).to(self.device)
                self.hidden_layers.append(hidden_layer)
            out_layer = nn.Linear(hidden_layer_dims[-1], action_size).to(self.device)
            self.hidden_layers.append(out_layer)
        else: 
            self.fc1 = nn.Linear(state_size, action_size).to(self.device)

        # self = self.to(self.device)
        
    def forward(self, x):
        if len(self.hidden_layers) > 0:
            x = F.relu(self.fc1(x))
            for layer_ix in range(len(self.hidden_layers)-1):
                x = F.relu(self.hidden_layers[layer_ix](x))
            
            x = F.softmax(self.hidden_layers[-1](x), dim=1)
        else:
            x = F.softmax(self.fc1(x), dim=1)
        
        return x
    
    def act(self, state):
        if type(state) != torch.Tensor:
            state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        probs = self.forward(state)#.cpu()
        m = Categorical(probs)
        action = m.sample()
        return action.item(), m.log_prob(action)



def reinforce_algorithm(env, policy, optimizer, n_training_episodes, max_t, gamma, print_every):
    # Help us to calculate the score during the training
    scores_deque = deque(maxlen=100)
    scores = []
    avg_reward = 0.
    # Line 3 of pseudocode
    pbar = tqdm(range(n_training_episodes))
    for i_episode in pbar:
        saved_log_probs = []
        rewards = []
        state = env.reset()
        # Line 4 of pseudocode
        for t in range(max_t):
            action, log_prob = policy.act(state)
            saved_log_probs.append(log_prob)
            state, reward, done, _ = env.step(action)
            rewards.append(reward)
            if done:
                break 
        scores_deque.append(sum(rewards))
        scores.append(sum(rewards))
        
        # Line 6 of pseudocode: calculate the return
        ## Here, we calculate discounts for instance [0.99^1, 0.99^2, 0.99^3, ..., 0.99^len(rewards)]
        discounts = [gamma**i for i in range(len(rewards)+1)]
        ## We calculate the return by sum(gamma[t] * reward[t]) 
        R = sum([a*b for a,b in zip(discounts, rewards)])
        
        # Line 7:
        policy_loss = []
        for log_prob in saved_log_probs:
            policy_loss.append(-log_prob * R)
        policy_loss = torch.cat(policy_loss).sum()
        
        # Line 8:
        optimizer.zero_grad()
        policy_loss.backward()
        optimizer.step()
        f1 = f1_score(env.target_history, env.prediction_history, pos_label="good")
        acc = accuracy_score(env.target_history, env.prediction_history)

        pbar.set_description(f"F1 Score: {f1:.2f}, accuracy: {acc:.2f} \t Average Score: {np.mean(scores_deque):.2f}")
            
        
    return scores

def evaluate_agent(env, max_steps, n_eval_episodes, policy):

    """
    Evaluate the agent for ``n_eval_episodes`` episodes and returns average reward and std of reward.
    :param env: The evaluation environment
    :param n_eval_episodes: Number of episode to evaluate the agent
    :param policy: The Reinforce agent
    """
    episode_rewards = []
    for episode in range(n_eval_episodes):
        state = env.reset()
        step = 0
        done = False
        total_rewards_ep = 0
    
        for step in range(max_steps):
            action, _ = policy.act(state)
            new_state, reward, done, info = env.step(action)
            total_rewards_ep += reward
            
            if done:
                break
            state = new_state
            
        episode_rewards.append(total_rewards_ep)

    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)

    return mean_reward, std_reward

def evaluate_on_clf(env, policy, pos_label: str = None):
    cnt = 0
    state = env.reset()
    while cnt < env.pool.n_samples:
        action, _ = policy.act(state)
        new_state, reward, done, info = env.step(action)
        if info["action"] not in ["<previous>", "<next>"]:
            cnt += 1
            print(f"Sample no: {cnt}", end="\r", flush=True)
        
        state = new_state
    
    preds = env.get_prediction_history()
    targets = env.get_target_history()
    acc = accuracy_score(targets, preds)
    f1 = f1_score(targets, preds, pos_label=pos_label)
    prec = precision_score(targets, preds, pos_label=pos_label)
    rec = recall_score(targets, preds, pos_label=pos_label)

    return acc, f1, prec, rec