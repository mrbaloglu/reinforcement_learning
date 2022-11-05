import sys
sys.path.append("/Users/emrebaloglu/Documents/RL/basic_reinforcement_learning")

import numpy as np
import pandas as pd
import NLP_utils.preprocessing as nlp_processing

import torch
from torch.utils.data import Dataset, DataLoader

from typing import List

class RTDataPoolWithWord2Vec(Dataset):
    vocab_size: int
    n_samples: int
    samples: np.ndarray
    labels: np.ndarray
    max_len: int

    def __init__(self, data_path, window_size: int, use_bert_tokens: bool = False):

        
        if use_bert_tokens:
            raise NotImplementedError
        else:
            data = pd.read_csv(data_path)
            data.columns = ['label', 'review']

            data_tkn = nlp_processing.tokenize_data(data, ["review"], preprocess=True)
            data_sqn = nlp_processing.pad_tokenized_data(data_tkn, ["review_tokenized"])

            self.max_len = len(data_sqn["review_tokenized"][0])
            self.n_samples = len(data_sqn)
            self.window_size = window_size

            assert window_size < self.max_len, \
                f"Number of words  to be read in each step (window_size) should be smaller than maximum sentence length, got window_size: {window_size}, max_len: {self.max_len}."
            
            self.samples = torch.from_numpy(np.stack(data_sqn["review_tokenized"].copy().values))
            
            pad_size = self.window_size - (self.max_len % self.window_size)
            if pad_size > 0:
                pad_m = np.zeros((self.n_samples, pad_size))
                self.samples = np.concatenate((self.samples, pad_m), axis=1)
                self.max_len = self.samples.shape[1]

            self.labels = data_sqn["label"].copy().values
    
    def create_episode(self, idx: int = None) -> List[np.ndarray]:
        if idx == None:
            idx = np.random.randint(self.n_samples)
        sample = self.samples[idx]
        label = self.labels[idx]
        states = np.split(sample, self.max_len / self.window_size)
        return states, label
    def __get_

if __name__ == "__main__":
    pool = RTDataPoolWithWord2Vec("NLP_datasets/rt-polarity-full.csv", 45)
    states, label = pool.create_episode()
    print(states)
    print(label)
        

    



