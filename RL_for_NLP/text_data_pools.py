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

from RL_for_NLP.observation import Observation, BertObservation



class PartialReadingDataPoolWithWord2Vec:
    
    def __init__(self, data: pd.DataFrame, text_col: str, tokenized_col: str, target_col: str, window_size: int, **kwargs):
        """Create a data pool for partial reading. Given a dataframe with processed, tokenized text inputs, the pool creates
        episodes for partial reading. (outputs chunks of text in selected length with label)

        Args:
            data (pd.DataFrame): Processed dataframe. Must include text column, tokenized text column, encoded label column, 
                                and label name column. (e.g: -------------dataframe-----------------
                                                             | text_col | tokens | label | label_str |
                                                             | good job |[25, 79]|   1   |  positive |
                                                             -----------------------------------------)
            text_col (str): name of the text column
            tokenized_col (str): name of the tokenized column
            target_col (str): name of the target column (dataframe must contain a column named 'target_col_str' 
                                having the names of the labels.)
            window_size (int): Number of words in each state to be created for an environment.
        """
        assert target_col + "_str" in data.columns, \
            f"Dataframe must contain a column named {target_col}_str having the names of the labels."
        
        self.max_len = len(data[tokenized_col][0])
        self.window_size = window_size  
        self.samples = []
        assert window_size < self.max_len, \
            f"Number of words  to be read in each step (window_size) should be smaller than maximum sentence length, got window_size: {window_size}, max_len: {self.max_len}."
        
        self.possible_actions = list(data[target_col + "_str"].unique())
        vecs = np.stack(data[tokenized_col].copy().values).astype(np.int32)
        self.n_samples = len(vecs)
        pad_size = self.window_size - (self.max_len % self.window_size)
        if pad_size > 0:
            pad_m = np.zeros((self.n_samples, pad_size))
            vecs = np.concatenate((vecs, pad_m), axis=1)
            self.max_len = vecs.shape[1]
        
        
        for j in range(vecs.shape[0]):
            sample_str = data[text_col][j]
            sample_vecs = np.split(vecs[j], self.max_len / self.window_size)
            label_enc = data[target_col][j]
            label_str = data[target_col + "_str"][j]
            obs = Observation(sample_str, sample_vecs, label_str, label_enc)
            self.samples.append(obs)
        
        
        # self.vecs = torch.from_numpy(self.vecs).float()
        self.labels = data[target_col].copy().values.astype(np.int32)# torch.from_numpy(data[target_col].copy().values).int()
            
    def create_episode(self, idx: int = None): # -> Tuple[List[torch.Tensor], torch.Tensor]:
        if idx == None:
            idx = np.random.randint(self.n_samples)

        idx = idx % self.n_samples

        return self.samples[idx]


    """def __getitem__(self, idx) -> Tuple[List[torch.Tensor], torch.Tensor]:
        states, label = self.create_episode(idx)
        return states, label"""
    
    def __len__(self) -> int:
        return self.n_samples
    

class PartialReadingDataPoolWithBertTokens:
    
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
                (df must contain ext_col_bert_input_ids, text_col_bert_token_type_ids, text_col_bert_attention_mask as columns as well)
            target_col (str): name of the target column (dataframe must contain a column named 'target_col_str' 
                                having the names of the labels.)
            window_size (int): Number of words in each state to be created for an environment.
        """
        required_cols = [text_col + "_bert_input_ids", text_col + "_bert_token_type_ids", text_col + "_bert_attention_mask"]
        assert set(required_cols) < set(data.columns), \
            f"Dataframe must contain columns named {required_cols}, but at least one of them is missing."

        assert target_col + "_str" in data.columns, \
            f"Dataframe must contain a column named {target_col}_str for label meanings."

        self.max_len = len(data[text_col+"_bert_input_ids"][0])
        print(f"Maximum sentence length in pool: {self.max_len}")
        self.window_size = window_size  
        self.samples = []
        assert window_size < self.max_len, \
            f"Number of words  to be read in each step (window_size) should be smaller than maximum sentence length, got window_size: {window_size}, max_len: {self.max_len}."
        
        self.possible_actions = list(data[target_col + "_str"].unique())

        input_id_vecs = np.stack(data[text_col + "_bert_input_ids"].copy().values).astype(np.int32)
        token_type_vecs = np.stack(data[text_col + "_bert_token_type_ids"].copy().values).astype(np.int32)
        attn_mask_vecs = np.stack(data[text_col + "_bert_attention_mask"].copy().values).astype(np.int32)
        
        self.n_samples = len(input_id_vecs)
        pad_size = self.window_size - (self.max_len % self.window_size)
       
        if pad_size > 0:
            pad_m = np.zeros((self.n_samples, pad_size))
            input_id_vecs = np.concatenate((input_id_vecs, pad_m), axis=1)
            token_type_vecs = np.concatenate((token_type_vecs, pad_m), axis=1)
            attn_mask_vecs = np.concatenate((attn_mask_vecs, pad_m), axis=1)
            self.max_len = input_id_vecs.shape[1]
        
        

        for j in range(input_id_vecs.shape[0]):
            sample_str = data[text_col][j]
            sample_input_id_vecs = np.split(input_id_vecs[j], self.max_len / self.window_size)
            sample_token_type_vecs = np.split(token_type_vecs[j], self.max_len / self.window_size)
            sample_attn_mask_vecs = np.split(attn_mask_vecs[j], self.max_len / self.window_size)
            label_enc = data[target_col][j]
            label_str = data[target_col + "_str"][j]
            obs = BertObservation(sample_str, sample_input_id_vecs, sample_token_type_vecs, sample_attn_mask_vecs, label_str, label_enc)
            self.samples.append(obs)
        
        
        # self.vecs = torch.from_numpy(self.vecs).float()
        self.labels = data[target_col].copy().values.astype(np.int32)# torch.from_numpy(data[target_col].copy().values).int()
            
    def create_episode(self, idx: int = None): # -> Tuple[List[torch.Tensor], torch.Tensor]:
        if idx == None:
            idx = np.random.randint(self.n_samples)

        idx = idx % self.n_samples

        return self.samples[idx]


    """def __getitem__(self, idx) -> Tuple[List[torch.Tensor], torch.Tensor]:
        states, label = self.create_episode(idx)
        return states, label"""
    
    def __len__(self) -> int:
        return self.n_samples


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
    # data_train = nlp_processing.openDfFromPickle("NLP_datasets/RT_Polarity/rt-polarity-train.pkl")
    # print(data_train.head())
    # pool = PartialReadingDataPoolWithWord2Vec(data_train, "review", "review_tokenized", "label", 16)

    # ix = np.random.randint(len(pool))
    # obs = pool.create_episode(ix)
    # print(obs)
    # print(obs.get_sample_vecs())
    # print(obs.get_label_enc())
    ######################################################################

    ############## pool with bert tokens #################################
    # data_train = nlp_processing.openDfFromPickle("NLP_datasets/RT_Polarity/rt-polarity-train-bert.pkl")
    # print(data_train.head())

    # pool = PartialReadingDataPoolWithBertTokens(data_train, "review", "label", "good", 11)
    # ix = np.random.randint(len(pool))
    # obs = pool.create_episode(ix)
    # print(obs)
    # print(obs.get_sample_input_id_vecs())
    # print(obs.get_sample_token_type_vecs())
    # print(obs.get_sample_attn_mask_vecs())
    # print(obs.get_label_enc())
    # print(obs.get_label_str())

    pool = SimpleSequentialDataPool(1000, 10, 2)
    obs = pool.create_episode()
    print(obs)

