from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from typing import List
import copy
import numpy as np


class PartialReadingRewardFunctionF1:
    """
    Computes F1 score between predicted and target labels
    It is a step-wise reward function
    Reward at time step t = score(t) - score(t-1)
    """
    def __init__(self, pos_label: str, label_list: List[str]):
        self.pos_label = pos_label
        self.label_list = label_list

    def __call__(self, action: str, target, prediction_history, prev_targets: List[str], exploration_discount: float = 0.) -> float:

        if action in self.label_list:
            # get current action sequence
            current_pred_history = copy.deepcopy(prediction_history)
            current_pred_history.append(action)

            current_targets = copy.deepcopy(prev_targets)
            current_targets.append(target)

            
            # step reward as change in the scores
            # as good actions lead to increase in the scores
            """print("-"*25)
            print(prev_targets)
            print(prediction_history)
            print("-"*25)"""
            previous_score = 0.
            if len(prev_targets) > 0:
                previous_score = f1_score(prev_targets, prediction_history, pos_label=self.pos_label, zero_division=0)
            """print("="*25)
            print(current_targets)
            print(current_pred_history)
            print("="*25)"""
            current_score = f1_score(current_targets, current_pred_history, pos_label=self.pos_label, zero_division=0)
            reward = current_score - previous_score

            return reward
        # give negative feedback when reread, previous or next, here curiosity may be applied
        else:
            return 0#-1e-2*(np.log2(exploration_discount))

class PartialReadingRewardFunctionAccuracy:
    """
    Computes accuracy score between predicted and target labels
    It is a step-wise reward function
    Reward at time step t = score(t) - score(t-1)
    """
    def __init__(self, label_list: List[str]):
        self.label_list = label_list

    def __call__(self, action: str, target: str, prediction_history: List[str], prev_targets: List[str], exploration_discount: float = 0.) -> float:

        if action in self.label_list:
            """# get current action sequence
            current_pred_history = copy.deepcopy(prediction_history)
            current_pred_history.append(action)

            current_targets = copy.deepcopy(prev_targets)
            current_targets.append(target)

            
            # step reward as change in the scores
            # as good actions lead to increase in the scores
            
            previous_score = 0.
            if len(prev_targets) > 0:
                previous_score = accuracy_score(prev_targets, prediction_history)
           
            current_score = accuracy_score(current_targets, current_pred_history)
            reward = current_score - previous_score
        """

            reward = int(action == target)
            return reward
        # give negative feedback when reread, previous or next, here curiosity may be applied
        else:
            return -0.1*(np.log2(exploration_discount))
