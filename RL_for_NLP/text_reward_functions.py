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

    """def __call__(self, action: str, target, prediction_history, prev_targets: List[str], exploration_discount: float = 0.) -> float:

        if action in self.label_list:
            # get current action sequence
            current_pred_history = copy.deepcopy(prediction_history)
            current_pred_history.append(action)

            current_targets = copy.deepcopy(prev_targets)
            current_targets.append(target)

            
            # step reward as change in the scores
            # as good actions lead to increase in the scores
            
            previous_score = 0.
            if len(prev_targets) > 0:
                previous_score = f1_score(prev_targets, prediction_history, pos_label=self.pos_label, zero_division=0)
            
            current_score = f1_score(current_targets, current_pred_history, pos_label=self.pos_label, zero_division=0)
            reward = current_score - previous_score

            return reward
        # give negative feedback when reread, previous or next, here curiosity may be applied
        else:
            return  -0.1*(np.log2(exploration_discount))"""

    def __call__(self, action: str, target: str, confusion_matrix: np.ndarray, prev_reward, exploration_discount: float = 0.) -> float:
        if action in self.label_list:
            action_ix = self.label_list.index(action)
            target_ix = self.label_list.index(target)
            confusion_matrix[action_ix, target_ix] += 1
            precisions = np.zeros((confusion_matrix.shape[0],))  # per class precision scores
            recalls = np.zeros((confusion_matrix.shape[0],)) # per class recall scores
            f1s = np.zeros((confusion_matrix.shape[0],)) # per class f1 scores
            for j in range(confusion_matrix.shape[0]):
                if np.sum(confusion_matrix[j, :]) != 0:
                    precisions[j] = confusion_matrix[j, j] / np.sum(confusion_matrix[j, :])
                    
                if np.sum(confusion_matrix[:, j]) != 0:
                    recalls[j] = confusion_matrix[j, j] / np.sum(confusion_matrix[:, j])
                
                if precisions[j] + recalls[j] != 0:
                    f1s[j] = 2*precisions[j]*recalls[j] / (precisions[j]+recalls[j])
        
                
            macro_f1 = np.average(f1s)
            return macro_f1, confusion_matrix
        else:
            return -0.0*(np.log2(exploration_discount)), confusion_matrix
        
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
            if reward == 0:
                reward = -1
            return reward
        # give negative feedback when reread, previous or next, here curiosity may be applied
        else:
            return 0.
