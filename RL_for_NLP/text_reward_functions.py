from typing import List, Dict
import numpy as np
from abc import ABC, abstractmethod

def calculate_stats_from_cm(confusion_matrix: np.ndarray, macro_avg: bool = True) -> Dict[str, float]:
    """Calculate accuracy, precision, recall and F1 Score from a given confusion matrix.
       By default, macro avg. of precision, recall and F1 Score is calculated.
    Args:
        confusion_matrix (np.ndarray): Confusion matrix.
        macro_avg (bool): Whether to calculate macro avg. or not. Set it false to calculate micro avg.
    Returns:
        Dict[str, float]: Dictionary of stats -- {"accuracy": acc_val, "precision": prec_val, "recall": rec_val, "f1": f1_val}
    """
    if macro_avg:
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
        macro_precision = np.average(precisions)
        macro_recall = np.average(recalls)
        accuracy = np.sum(np.diagonal(confusion_matrix)) / np.sum(confusion_matrix)

        return {"accuracy": accuracy, "precision": macro_precision, "recall": macro_recall, "f1": macro_f1}
    else: 
        raise NotImplementedError

class PartialReadingReward(ABC):
    def __init__(self, label_list: List[str]):
        self.label_list = label_list

    def update_cm(self, action: str, target: str, confusion_matrix: np.ndarray) -> np.ndarray:
        assert action in self.label_list, \
            f"Given action: {action} is not a prediction."
        
        action_ix = self.label_list.index(action)
        target_ix = self.label_list.index(target)
        confusion_matrix[action_ix, target_ix] += 1

        return confusion_matrix

    @abstractmethod
    def  __call__(self, action: str, target: str, confusion_matrix: np.ndarray, expolarition_discount: float = 0.) -> float:
        raise NotImplementedError

class PartialReadingRewardF1(PartialReadingReward):
    """
    Computes F1 score between predicted and target labels
    It is a step-wise reward function
    Reward at time step t = score(t) - score(t-1) (handled in environment code)
    """
    def __call__(self, action: str, target: str, confusion_matrix: np.ndarray, exploration_discount: float = 0.) -> float:
        if action in self.label_list:
            confusion_matrix = super().update_cm(action, target, confusion_matrix)
            macro_f1 = calculate_stats_from_cm(confusion_matrix)["f1"]
            # tm = 1
            # if action != target:
            #     tm = -1.25

            return  macro_f1, confusion_matrix 
        else:
            return -0.0, confusion_matrix # -0.0*(np.log2(exploration_discount)), confusion_matrix
        
class PartialReadingRewardAccuracy(PartialReadingReward):
    """
    Computes accuracy score between predicted and target labels
    It is a step-wise reward function
    Reward at time step t = score(t) - score(t-1) (handled in environment code)
    """
    
    def __call__(self, action: str, target: str, confusion_matrix: np.ndarray, exploration_discount: float = 0.) -> float:

        if action in self.label_list:
            confusion_matrix = super().update_cm(action, target, confusion_matrix)
            accuracy = calculate_stats_from_cm(confusion_matrix)["accuracy"]
            
            return accuracy, confusion_matrix
        else:
            return -0.0*(np.log2(exploration_discount)), confusion_matrix

class PartialReadingRewardPrecision(PartialReadingReward):
    """
    Computes precision score between predicted and target labels
    It is a step-wise reward function
    Reward at time step t = score(t) - score(t-1) (handled in environment code)
    """
    
    def __call__(self, action: str, target: str, confusion_matrix: np.ndarray, exploration_discount: float = 0.) -> float:

        if action in self.label_list:
            confusion_matrix = super().update_cm(action, target, confusion_matrix)
            precision = calculate_stats_from_cm(confusion_matrix)["precision"]
            
            return precision, confusion_matrix
        else:
            return -0.0*(np.log2(exploration_discount)), confusion_matrix

class PartialReadingRewardRecall(PartialReadingReward):
    """
    Computes recall score between predicted and target labels
    It is a step-wise reward function
    Reward at time step t = score(t) - score(t-1) (handled in environment code)
    """
    
    def __call__(self, action: str, target: str, confusion_matrix: np.ndarray, exploration_discount: float = 0.) -> float:

        if action in self.label_list:
            confusion_matrix = super().update_cm(action, target, confusion_matrix)
            recall = calculate_stats_from_cm(confusion_matrix)["recall"]
            
            return recall, confusion_matrix
        else:
            return -0.0*(np.log2(exploration_discount)), confusion_matrix

class PartialReadingRewardScore(PartialReadingReward):
    def __call__(self, action: str, target: str, confusion_matrix: np.ndarray, exploration_discount: float = 0) -> float:
        if action in self.label_list:
            confusion_matrix = super().update_cm(action, target, confusion_matrix)
            tmp = int(action == target)
            if tmp == 0:
                tmp = -1.25
            return tmp, confusion_matrix
        else:
            return 0.01 * exploration_discount, confusion_matrix
    