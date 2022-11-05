from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

import numpy as np
from collections import deque

from typing import List, Union



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

class RNN_Baseline_Policy(nn.Module):
    def __init__(self,
                 vocab_size: int,
                 input_dim: int,
                 out_num_classes: int,
                 embed_dim: int = 5,
                 rnn_type: str = "gru",
                 rnn_hidden_size: int = 2,
                 rnn_hidden_out: int = 2,
                 rnn_bidirectional: bool = True,
                 units: int = 50):
        """
        RNN Class for text classification with RL.
        Arguments:
            vocab_size -- specifies the size of the vocabulary to be used in word embeddings.
            input_dim -- specifies the dimension of the features (n_samples, max_sentence_length (input_dim)).
            output_dim -- specifies the number of possible actions for the agent. 

        Keyword Arguments:
            embed_dim -- embed_dim specifies the embedding dimension for the categorical part of the input. (default: {5})
            rnn_type -- specifies the type of the recurrent layer for word embeddings. (default: {"gru"})
            rnn_hidden_size -- specifies the number of stacked recurrent layers. (default: {2})
            rnn_hidden_out -- specifies number of features in the hidden state h of recurrent layer. (default: {2})
            rnn_bidirectional -- specifies whether the recurrent layers be bidirectional. (default: {True})
            units -- specifies the number of neurons in the hidden layers. (default: {50})

        """
        super(RNN_Baseline_Policy, self).__init__()

        self.embed_dim = embed_dim
        self.out_num_classes = out_num_classes

        self.embed_enc = nn.Embedding(vocab_size, embed_dim, max_norm=True)
        self.rnn = nn.GRU(embed_dim, rnn_hidden_out, rnn_hidden_size,
                          bidirectional=rnn_bidirectional, batch_first=True)
        if rnn_type == "lstm":
            self.rnn = nn.LSTM(embed_dim, rnn_hidden_out, rnn_hidden_size,
                               bidirectional=rnn_bidirectional, batch_first=True)
        elif rnn_type != "gru":
            raise ValueError("The argument rnn_type must be 'gru' or 'lstm'!")

        rnn_out_dim = rnn_hidden_out * input_dim
        if rnn_bidirectional:
            rnn_out_dim *= 2

        self.flat = nn.Flatten()
        self.fc1 = nn.Linear(rnn_out_dim, units)
        self.fc2 = nn.Linear(units, units)
        self.fc3 = nn.Linear(units, out_num_classes)

    def forward(self, x: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """
        Model prediction function.

        Args:
            x (Union[np.ndarray, torch.Tensor]): data

        Returns:
            torch.Tensor: predictions.
        """

        # type check
        if type(x) != torch.Tensor:
            x = torch.Tensor(x)

        # reshape when there is only one sample
        if len(x.shape) == 1:
            x = torch.unsqueeze(x, 0)
        
        x_c = self.embed_enc(x.int())
        x_c, _ = self.rnn(x_c)
        x_c = self.flat(x_c)

        x = F.relu(self.fc1(x_c))
        x = F.relu(self.fc2(x))
        x = torch.softmax(self.fc3(x), dim=1)

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
    # Line 3 of pseudocode
    for i_episode in range(1, n_training_episodes+1):
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
        
        if i_episode % print_every == 0:
            print('Episode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)))
        
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