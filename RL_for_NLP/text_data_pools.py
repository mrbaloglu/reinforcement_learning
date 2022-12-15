import sys
# sys.path.append("/Users/emrebaloglu/Documents/RL/basic_reinforcement_learning") # macos
sys.path.append("C:\\Users\\mrbal\\Documents\\NLP\\RL\\basic_reinforcement_learning")

import numpy as np
import pandas as pd
import NLP_utils.preprocessing as nlp_processing

import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from typing import List, Tuple
from abc import ABC, abstractmethod
from dataclasses import dataclass
from tqdm import tqdm

from RL_for_NLP.observation import Observation, BertObservation

class PartialReadingDataPool(ABC):
    def __init__(self, data: pd.DataFrame, text_col: str, target_col: str, window_size: int, **kwargs):
        self.window_size = window_size
        self.n_samples = len(data)
        self.samples = []
        self.labels = data[target_col].copy().values.astype(np.int32)
        self.max_len = -1 # will be updated with populate_samples function
        self.vocab_size = -1 # will be updated with populate_samples function
        self.possible_actions = list(data[target_col + "_str"].unique())
    
    def create_episode(self, idx: int = None): # -> Tuple[List[torch.Tensor], torch.Tensor]:
        if idx == None:
            idx = np.random.randint(self.n_samples)

        idx = idx % self.n_samples

        return self.samples[idx]

    # populate the samples and calculate the maximum sentence length in pool
    @abstractmethod
    def populate_samples(self, data: pd.DataFrame, text_col: str, target_col: str, window_size: int, **kwargs):
        raise NotImplementedError
    
    def __len__(self) -> int:
        return self.n_samples
    



class PartialReadingDataPoolWithTokens(PartialReadingDataPool):
    
    def __init__(self, data: pd.DataFrame, text_col: str, target_col: str, window_size: int, **kwargs):
        """Create a data pool for partial reading. Given a dataframe with processed, tokenized text inputs, the pool creates
        episodes for partial reading. (outputs chunks of text in selected length with label)

        Args:
            data (pd.DataFrame): Processed dataframe. Must include text column, tokenized text column, encoded label column, 
                                and label name column. (e.g: -------------dataframe-----------------
                                                             | text_col | tokens | label | label_str |
                                                             | good job |[25, 79]|   1   |  positive |
                                                             -----------------------------------------)
            text_col (str): name of the text column
            -- data must also contain a column named {text_col}_tokenized, containing arrays of the same length (tokenized sentences)
            target_col (str): name of the target column (dataframe must contain a column named 'target_col_str' 
                                having the names of the labels.)
            window_size (int): Number of words in each state to be created for an environment.
        """
        assert target_col + "_str" in data.columns, \
            f"Dataframe must contain a column named '{target_col}_str' having the names of the labels."
        
        assert text_col + "_tokenized" in data.columns, \
            f"Dataframe must contain a column named '{text_col}_tokenized' having the names of the labels."
        
        super().__init__(data, text_col, target_col, window_size, kwargs=kwargs)

        self.populate_samples(data, text_col, target_col, window_size, kwargs=kwargs)

    # populate each sample with token vectors and corresponding labels.
    def populate_samples(self, data: pd.DataFrame, text_col: str, target_col: str, window_size: int, **kwargs):

        self.max_len = len(data[text_col + "_tokenized"][0])
        print(f"Maximum sentence length in pool: {self.max_len}")
        assert window_size < self.max_len, \
            f"Number of words  to be read in each step (window_size) should be smaller than maximum sentence length, got window_size: {window_size}, max_len: {self.max_len}."
        
        
        vecs = np.stack(data[text_col + "_tokenized"].copy().values).astype(np.int32)
        self.n_samples = len(vecs)
        self.vocab_size = np.max(vecs) + 1
        pad_size = self.window_size - (self.max_len % self.window_size)

        if pad_size > 0:
            pad_m = np.zeros((self.n_samples, pad_size))
            vecs = np.concatenate((vecs, pad_m), axis=1)
            self.max_len = vecs.shape[1]
        
        
        for j in tqdm(range(vecs.shape[0]), desc="Padding the data and populating samples..."):
            sample_str = data[text_col][j]
            sample_vecs = np.split(vecs[j], self.max_len / self.window_size)
            label_enc = data[target_col][j]
            label_str = data[target_col + "_str"][j]
            obs = Observation(sample_str, sample_vecs, label_str, label_enc)
            self.samples.append(obs)
        
class PartialReadingDataPoolWithBertTokens(PartialReadingDataPool):
    
    def __init__(self, data: pd.DataFrame, text_col: str, target_col: str, window_size: int, **kwargs):
        """Create a data pool for partial reading. Given a dataframe with processed, bert-tokenized text inputs, the pool creates
        episodes for partial reading. (outputs chunks of text in selected length with label)

        Args:
            data (pd.DataFrame): Processed dataframe. Must include text column, tokenized text column, encoded label column, 
                                and label name column. 
            (e.g: -------------dataframe-----------------
                | text_col | text_col_bert_input_ids | text_col_bert_token_type_ids | text_col_bert_attention_mask | label | label_str |
                | good job | [25, 79, ..., 0, 0]     |  [1, 1, 1, ..., 0, 0]        |  [0, 0, ..., 0]              |   1   |  positive |
                -----------------------------------------)
            text_col (str): name of the text column 
                (df must contain columns named {text_col}_bert_input_ids, {text_col}_bert_token_type_ids, {text_col}_bert_attention_mask)
            target_col (str): name of the target column (dataframe must contain a column named 'target_col_str' 
                                having the names of the labels.)
            window_size (int): Number of words in each state to be created for an environment.
        """
        required_cols = [text_col + "_bert_input_ids", text_col + "_bert_token_type_ids", text_col + "_bert_attention_mask"]
        assert set(required_cols) < set(data.columns), \
            f"Dataframe must contain columns named {required_cols}, but at least one of them is missing."

        assert target_col + "_str" in data.columns, \
            f"Dataframe must contain a column named {target_col}_str for label meanings."
        
        super().__init__(data, text_col, target_col, window_size, kwargs=kwargs)
        self.populate_samples(data, text_col, target_col, window_size, kwargs=kwargs)
    
    # populate the samples with bert tokens, attention masks and labels
    def populate_samples(self, data: pd.DataFrame, text_col: str, target_col: str, window_size: int, **kwargs):

        self.max_len = len(data[text_col+"_bert_input_ids"][0])
        print(f"Maximum sentence length in pool: {self.max_len}")
        assert window_size <= self.max_len, \
            f"Number of words  to be read in each step (window_size) should be smaller than maximum sentence length, got window_size: {window_size}, max_len: {self.max_len}."
        

        input_id_vecs = np.stack(data[text_col + "_bert_input_ids"].copy().values).astype(np.int32)
        attn_mask_vecs = np.stack(data[text_col + "_bert_attention_mask"].copy().values).astype(np.int32)
        
        self.n_samples = len(input_id_vecs)
        self.vocab_size = np.max(input_id_vecs)
        pad_size = self.window_size - (self.max_len % self.window_size)
       
        if pad_size > 0:
            pad_m = np.zeros((self.n_samples, pad_size))
            input_id_vecs = np.concatenate((input_id_vecs, pad_m), axis=1)
            attn_mask_vecs = np.concatenate((attn_mask_vecs, pad_m), axis=1)
            self.max_len = input_id_vecs.shape[1]
        
        

        for j in tqdm(range(input_id_vecs.shape[0]), desc="Padding the data and populating samples..."):
            sample_str = data[text_col][j]
            sample_input_id_vecs = np.split(input_id_vecs[j], self.max_len / self.window_size)
            sample_attn_mask_vecs = np.split(attn_mask_vecs[j], self.max_len / self.window_size)
            label_enc = data[target_col][j]
            label_str = data[target_col + "_str"][j]
            obs = BertObservation(sample_str, sample_input_id_vecs, sample_attn_mask_vecs, label_str, label_enc)
            self.samples.append(obs)



class SimpleSequentialDataPool:
    
    def __init__(self, n_samples, n_features, window_size: int, **kwargs):
        self.n_samples = n_samples
        self.n_features = n_features
        data = np.zeros((n_samples, n_features))
        labels = np.zeros((n_samples))
        for j in range(data.shape[0]):
            ix = np.random.randint(data.shape[1])
            val = np.random.choice([-1, 1])
            data[j, ix] = val
            if val == 1:
                labels[j] = 1
        
        self.data = data
        self.labels = labels
        self.window_size = window_size  
        self.samples = []
        assert window_size < self.n_features, \
            f"Number of words  to be read in each step (window_size) should be smaller than maximum sentence length, got window_size: {window_size}, max_len: {self.max_len}."
        
        self.possible_actions = ["pos", "neg"]
        self.ix_to_str = {0: "neg", 1: "pos"}
        self.max_len = n_features
        
        pad_size = self.window_size - (self.max_len % self.window_size)
       
        if pad_size > 0:
            pad_m = np.zeros((self.n_samples, pad_size))
            self.data = np.concatenate((self.data, pad_m), axis=1)
            self.max_len = self.data.shape[1]
        
        

        for j in range(self.data.shape[0]):

            sample_data = np.split(self.data[j], self.max_len / self.window_size)
            self.samples.append(sample_data)
        
    def create_episode(self, idx: int = None): # -> Tuple[List[torch.Tensor], torch.Tensor]:
        if idx == None:
            idx = np.random.randint(self.n_samples)

        idx = idx % self.n_samples

        return self.samples[idx], self.ix_to_str[self.labels[idx]]


    """def __getitem__(self, idx) -> Tuple[List[torch.Tensor], torch.Tensor]:
        states, label = self.create_episode(idx)
        return states, label"""
    
    def __len__(self) -> int:
        return self.n_samples
 


if __name__ == "__main__":
    
    ############## pool with regular tokens ############################
    data_train = nlp_processing.openDfFromPickle("NLP_datasets/RT_Polarity/rt-polarity-train.pkl")
    print(data_train.head())
    pool = PartialReadingDataPoolWithTokens(data_train, "review", "label", 16)
    print(pool.vocab_size)
    ix = np.random.randint(len(pool))
    obs = pool.create_episode(ix)
    print(obs)
    print(obs.get_sample_vecs())
    print(obs.get_label_enc())
    ######################################################################

    ############## pool with bert tokens #################################
    # data_train = nlp_processing.openDfFromPickle("NLP_datasets/RT_Polarity/rt-polarity-train-bert.pkl")
    # print(data_train.head())

    # pool = PartialReadingDataPoolWithBertTokens(data_train, "review", "label", 11)
    # print(pool.vocab_size)
    # ix = np.random.randint(len(pool))
    # obs = pool.create_episode(ix)
    # print(obs)
    # print(obs.get_sample_input_id_vecs())
    # print(obs.get_sample_attn_mask_vecs())
    # print(obs.get_label_enc())
    # print(obs.get_label_str())
    

    # pool = SimpleSequentialDataPool(1000, 10, 2)
    # obs = pool.create_episode()
    # print(obs)

