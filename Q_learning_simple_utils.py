import numpy as np
import gym
from tqdm import tqdm
from typing import Tuple

def initialize_q_table(num_states: int, num_actions: int) -> np.ndarray:
    """
    Initialize a Q-table to be used for storing state-action values.

    Args:
        num_states (int): Number of possible states in the environment.
        num_actions (int): Number of possible actions that can be taken by the agent.

    Returns:
        np.ndarray: Initialized Q-table for the environment to zero-values for each state-action pair.
    """
    return np.zeros((num_states, num_actions))

def epsilon_greedy_policy(Qtable: np.ndarray, state: int, epsilon: float) -> int:
    """
    Choose an epsilon-greedy action given a state in a Q-table, with given epsilon value.

    Args:
        Qtable (np.ndarray): The table of values for state-action pairs.
        state (int): index of the current state where action to be taken.
        epsilon (float): expolarition probability (must be btw 0 and 1)

    Returns:
        int: action index
    """
    assert epsilon >= 0 and epsilon <= 1, \
        f"The value of epsilon must be in [0, 1], got {epsilon}."
    
    # Randomly generate a number between 0 and 1
    random_num = np.random.rand()
    # if random_num > greater than epsilon --> exploitation
    if random_num > epsilon:
      # Take the action with the highest value given a state
      # np.argmax can be useful here
      action = np.argmax(Qtable[state, :])
    # else --> exploration
    else:
      action = np.random.randint(0, len(Qtable[state]))# Take a random action

    return action

def greedy_policy(Qtable: np.ndarray, state: int) -> int:
    """
    Choose a greedy action given a state in a Q-table.

    Args:
        Qtable (np.ndarray): The table of values for state-action pairs.
        state (int): index of the current state where action to be taken.

    Returns:
        int: action index
    """
    # Exploitation: take the action with the highest state, action value
    action = np.argmax(Qtable[state])
    
    return action

def train_q_learning(n_training_episodes: int, min_epsilon: float, max_epsilon: float, decay_rate: float, 
                learning_rate: float, gamma: float, env: gym.Env, max_steps: int, Qtable: np.ndarray) -> np.ndarray:
    """
    Train an agent with Q-Learning algorithm.

    Args:
        n_training_episodes (int): Number of episodes for training.
        min_epsilon (float): Minimum value for epsilon (exploration prob.).
        max_epsilon (float): Maximum value for epsilon.
        decay_rate (float): Exponential decay rate for epsilon. (will decay through training)
        learning_rate (float): Learning rate of the algorithm.
        gamma (float): The discount rate. 
        env (gym.Env): The environment where the agent interacts.
        max_steps (int): Maximum steps that agent can take unless it goes into a terminal state.
        Qtable (np.ndarray): Q-table of the agent.

    Returns:
        np.ndarray: Updated Q-table, resulting with training.
    """

    assert decay_rate >= 0 and decay_rate <= 1, \
        f"The value of decay rate must be in [0, 1], got {decay_rate}." 
    
    assert learning_rate >= 0 and learning_rate <= 1, \
        f"The value of learning rate must be in [0, 1], got {learning_rate}."
    
    assert gamma >= 0 and gamma <= 1, \
        f"The value of gamma must be in [0, 1], got {gamma}."

    for episode in tqdm(range(n_training_episodes)):
        # Reduce epsilon (because we need less and less exploration)
        epsilon = min_epsilon + (max_epsilon - min_epsilon)*np.exp(-decay_rate*episode)
        # Reset the environment
        state = env.reset()
        step = 0
        done = False

        # repeat
        for step in range(max_steps):
            # Choose the action At using epsilon greedy policy
            action = epsilon_greedy_policy(Qtable, state, epsilon)

            # Take action At and observe Rt+1 and St+1
            # Take the action (a) and observe the outcome state(s') and reward (r)
            new_state, reward, done, info = env.step(action)

            # Update Q(s,a):= Q(s,a) + lr [R(s,a) + gamma * max Q(s',a') - Q(s,a)]
            Qtable[state][action] = Qtable[state][action] + learning_rate*(reward + gamma*np.max(Qtable[new_state]) - Qtable[state][action])

            # If done, finish the episode
            if done:
                break
            
            # Our state is the new state
            state = new_state
    return Qtable

def evaluate_agent(env: gym.Env, max_steps: int, n_eval_episodes: int, Q: np.ndarray, seed: list) -> Tuple[float, float]:
    """
    Evaluate the agent for ``n_eval_episodes`` episodes and returns average reward and std of reward.

    Args:
        env (gym.Env): The evaluation environment
        max_steps (int): Maximum steps that agent can take unless it goes into a terminal state.
        n_eval_episodes (int): Number of episode to evaluate the agent
        Q (np.ndarray): The Q-table
        seed (list): The evaluation seed array (for taxi-v3)

    Returns:
        Tuple[float, float]: mean of reward and std. dev. of rewards
    """
   
    episode_rewards = []
    for episode in tqdm(range(n_eval_episodes)):
        if seed:
            state = env.reset(seed=seed[episode])
        else:
            state = env.reset()
            step = 0
            done = False
            total_rewards_ep = 0
        
        for step in range(max_steps):
            # Take the action (index) that have the maximum expected future reward given that state
            action = np.argmax(Q[state][:])
            new_state, reward, done, info = env.step(action)
            total_rewards_ep += reward
            
            if done:
                break
            state = new_state
            
        episode_rewards.append(total_rewards_ep)
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)

    return mean_reward, std_reward