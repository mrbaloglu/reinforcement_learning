"""
File containing functions for preprocessing csv files with textual data.
"""

import numpy as np
import pandas as pd
import re
from tqdm.notebook import tqdm_notebook
from keras.preprocessing import text, sequence
from typing import Union
from tqdm import tqdm
from tqdm.notebook import  tqdm_notebook

def preprocess_text(sen: str) -> list:
    """
    Preprocess a text input.
    :param sen: Sentence, or textual input to be processed.
    :return: Processed text.
    """
    # Remove punctuations and numbers
    sentence = re.sub('[^a-zA-Z]', ' ', sen)

    # Single character removal
    sentence = re.sub(r"\s+[a-zA-Z]\s+", ' ', sentence)

    # Removing multiple spaces
    sentence = re.sub(r'\s+', ' ', sentence)

    # Sequentialize the text word-by-word and remove special characters ''.join(e for e in string if e.isalnum())
    sentence = " ".join(["".join(c for c in word if c.isalnum()) for word in sentence.split(" ")])

    return sentence

def process_df_texts(data: pd.DataFrame, keys: list) -> pd.DataFrame:
    """
    Function for cleaning texts from special characters, single characters, multiple spaces and punctuations
    and making all characters into lowercase.
    :param data: Dataframe in which the textual data to be processed.
    :param keys: Column names of the textual data.
    :return: Processed dataframe.
    """
    for key in keys:
        tqdm_notebook.pandas(desc=f"Applying text-processing on {key}")
        data[key] = data[key].apply(lambda x: preprocess_text(x))

    return data

def tokenize_data(data: pd.DataFrame, keys: list,
                  create_new_column: bool = True,
                  preprocess: bool = False) -> pd.DataFrame:
    """
    Function for tokenizing the textual input in a dataframe, given the columns.
    :param data: Dataframe containing the textual data to be tokenized.
    :param keys: Textual columns in the dataframe
    :param create_new_column: boolean parameter to decide
    whether to create a new column for tokenized texts.
    :param preprocess: parameter for whether preprocessing the data. Use True if dataframe is
    unprocessed.
    :return: Dataframe with tokenized texts
    """
    one_column = False  # will be true when only one column is tokenized, for list <-> str conversions of keys

    if preprocess:
        data = process_df_texts(data, keys)

    tokenizer = text.Tokenizer(lower=False)
    if len(keys) == 1:
        tokenizer.fit_on_texts(data[keys[0]].values)
    else:
        tokenizer.fit_on_texts(data[keys])
    print(f"Vocabulary size: {len(tokenizer.word_index) + 1}.")


    if create_new_column:
        for key in tqdm(keys):
            tokenized = tokenizer.texts_to_sequences(data[key].values)
            data[key +  "_tokenized"] = tokenized
    else:
        data[keys] = tokenizer.texts_to_sequences(data[keys])

        for key in tqdm(keys):
            tokenized = tokenizer.texts_to_sequences(data[key])
            data[key] = tokenized

    return data

def pad_tokenized_data(data: pd.DataFrame, keys: list) -> pd.DataFrame:
    """

    :param data: Dataframe containing the tokenized sequences to be padded.
    :param keys: List of tokenized columns of dataframe
    :return: Dataframe with padded sequences.
    """
    if len(keys) == 1:
        max_len = max(len(x) for x in data[keys[0]].values)
        print(f"On column {keys}, maximum sentence length is {max_len}.")
        sq = sequence.pad_sequences(data[keys[0]], maxlen=max_len, padding="post")
        data[keys[0]] = [x for x in sq]

    else:
        for key in keys:
            max_len = max(len(x) for x in data[key].values)
            print(f"On column {key}, maximum sentence length is {max_len}.")
            sq = sequence.pad_sequences(data[key])
            data[key] = [x for x in sq]

    return data
