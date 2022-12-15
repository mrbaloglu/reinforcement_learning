import torch
from tqdm import tqdm

import numpy as np
from collections import deque
from RL_for_NLP.text_reward_functions import calculate_stats_from_cm

from typing import List
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score



def reinforce_algorithm(env, policy, optimizer, n_training_episodes, max_t, gamma):
    # Help us to calculate the score during the training
    scores_deque = deque(maxlen=100)
    scores = []
    actions = []
    avg_reward = 0.
    n_next = 0
    n_pos = 0
    n_neg = 0
    possible_actions = env.pool.possible_actions
    stats = {}
    pbar = range(n_training_episodes)
    for i_episode in pbar:
        saved_log_probs = []
        rewards = []
        state, *_ = env.step(0) # go to the next sample
        
        for t in tqdm(range(max_t)):
            action, log_prob = policy.predict(state)
            action_str = env.action_space.ix_to_action(action)
            actions.append(action_str)
            if action_str == "<next>":
                n_next += 1
            elif action_str == "good":
                n_pos += 1
            else:
                n_neg += 1
            saved_log_probs.append(log_prob)
            state, reward, done, _ = env.step(action)
            rewards.append(reward)
            if done:
                break 
        scores_deque.append(sum(rewards))
        scores.append(sum(rewards))
        
        
        ## Here, we calculate discounts for instance [0.99^1, 0.99^2, 0.99^3, ..., 0.99^len(rewards)]
        discounts = [gamma**i for i in range(len(rewards)+1)]
        ## We calculate the return by sum(gamma[t] * reward[t]) 
        R = sum([a*b for a,b in zip(discounts, rewards)])
        
        
        policy_loss = []
        for log_prob in saved_log_probs:
            policy_loss.append(-log_prob * R)
        policy_loss = torch.cat(policy_loss).sum()
        
        
        optimizer.zero_grad()
        policy_loss.backward()
        optimizer.step()
        
        
        stats = calculate_stats_from_cm(env.confusion_matrix)

        print(f"Episode: {i_episode}, Stats: {stats}, {len(actions), n_pos, n_neg, n_next}")
            
        
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
            action, _ = policy.predict(state)
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
        action, _ = policy.predict(state)
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