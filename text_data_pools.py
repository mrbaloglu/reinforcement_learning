import sys
sys.path.append("/Users/emrebaloglu/Documents/RL/basic_reinforcement_learning")

import numpy as np
import pandas as pd
import NLP_utils.preprocessing as nlp_processing

import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from typing import List, Tuple
from abc import ABC, abstractmethod
from dataclasses import dataclass



class RTDataPoolWithWord2Vec:
    
    def __init__(self, data: pd.DataFrame, tokenized_col: str, target_col:str, window_size: int, **kwargs):

        self.max_len = len(data[tokenized_col][0])
        self.window_size = window_size  
        
        assert window_size < self.max_len, \
            f"Number of words  to be read in each step (window_size) should be smaller than maximum sentence length, got window_size: {window_size}, max_len: {self.max_len}."
        
        self.possible_actions = list(data[target_col + "_str"].unique())
        samples = np.stack(data[tokenized_col].copy().values).astype(np.int32)
        self.n_samples = len(samples)
        pad_size = self.window_size - (self.max_len % self.window_size)
        if pad_size > 0:
            pad_m = np.zeros((self.n_samples, pad_size))
            samples = np.concatenate((samples, pad_m), axis=1)
            self.max_len = samples.shape[1]
        
        self.samples = []
        for j in range(samples.shape[0]):
            self.samples.append(np.split(samples[j], self.max_len / self.window_size))
        
        
        # self.samples = torch.from_numpy(self.samples).float()
        self.labels = data[target_col].copy().values.astype(np.int32)# torch.from_numpy(data[target_col].copy().values).int()
            
    def create_episode(self, idx: int = None): # -> Tuple[List[torch.Tensor], torch.Tensor]:
        if idx == None:
            idx = np.random.randint(self.n_samples)

        idx = idx % self.n_samples
        states = self.samples[idx]
        label = self.labels[idx]

        return states, label


    """def __getitem__(self, idx) -> Tuple[List[torch.Tensor], torch.Tensor]:
        states, label = self.create_episode(idx)
        return states, label
    
    def __len__(self) -> int:
        return self.n_samples"""
    

if __name__ == "__main__":
    data_train = nlp_processing.openDfFromPickle("NLP_datasets/rt-polarity-train.pkl")
    data_train, tokenizer = nlp_processing.tokenize_data(data_train, ["review"], preprocess=True)
    print("n_words_1: ", len(tokenizer.word_index) + 1)
    data_train, max_len = nlp_processing.pad_tokenized_data(data_train, ["review_tokenized"])
    print(data_train.head())
    pool = RTDataPoolWithWord2Vec(data_train, "review_tokenized", "label", 16)

    tmp_states = None
    tmp_labels = None
    for i in range(1):
        tmp_states, tmp_labels = pool.create_episode(i)

    print(len(tmp_states))
    print(tmp_states[0].shape)
        

    



