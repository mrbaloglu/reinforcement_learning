from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from typing import List
import copy


class PartialReadingRewardFunctionF1:
    """
    Computes F1 score between predicted and target labels
    It is a step-wise reward function
    Reward at time step t = score(t) - score(t-1)
    """
    def __init__(self):
        pass

    def __call__(self, action: str, target, prediction_history, prev_targets: List[str]) -> float:

        if action == "good" or action == "bad":
            # get current action sequence
            current_pred_history = copy.deepcopy(prediction_history)
            current_pred_history.append(action)

            current_targets = copy.deepcopy(prev_targets)
            current_targets.append(target)

            # remove "next", "reread", "previous" to compute reward
            if "<next>" in current_pred_history:
                current_pred_history.remove("<next>")
            if "<reread>" in current_pred_history:
                current_pred_history.remove("<reread>")
            if "<previous>" in current_pred_history:
                current_pred_history.remove("<previous>")
            
            # step reward as change in the scores
            # as good actions lead to increase in the scores
            """print("-"*25)
            print(prev_targets)
            print(prediction_history)
            print("-"*25)"""
            previous_score = 0.
            if len(prev_targets) > 0:
                previous_score = f1_score(prev_targets, prediction_history, pos_label="good", zero_division=0)
            """print("="*25)
            print(current_targets)
            print(current_pred_history)
            print("="*25)"""
            current_score = f1_score(current_targets, current_pred_history, pos_label="good", zero_division=0)
            reward = current_score - previous_score

            return reward
        # give negative feedback when reread, previous or next
        else:
            return -1e-4
