"""
File containing functions for preprocessing csv files with textual data.
"""

import numpy as np
import pandas as pd
import re
from tqdm.notebook import tqdm_notebook
from keras.preprocessing import text, sequence
from typing import Union, List
from tqdm import tqdm
from tqdm.notebook import  tqdm_notebook
import pickle
import json
import transformers

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
                  preprocess: bool = False,
                  tokenizer: text.Tokenizer = None) -> pd.DataFrame:
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
    if tokenizer == None:
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

    return data, tokenizer

def pad_tokenized_data(data: pd.DataFrame, keys: list, max_len: int = 0) -> pd.DataFrame:
    """Pad the tokenized data samples with zeros.

    Args:
        data (pd.DataFrame): data to be padded
        keys (list): columns with tokenized samples
        max_len (int, optional): Length the resulting padded samples, 
                                if 0 automatically calculated as maximum sentence length in given data. 
                                Defaults to 0.

    Returns:
        pd.DataFrame: Dataframe with given columns padded.
    """
    if len(keys) == 1:
        if max_len == 0:
            max_len = max(len(x) for x in data[keys[0]].values)
        print(f"On column {keys}, maximum sentence length is {max_len}.")
        sq = sequence.pad_sequences(data[keys[0]], maxlen=max_len, padding="post")
        data[keys[0]] = [x for x in sq]

    else:
        for key in keys:
            if max_len == 0:
                max_len = max(len(x) for x in data[key].values)
            print(f"On column {key}, maximum sentence length is {max_len}.")
            sq = sequence.pad_sequences(data[key])
            data[key] = [x for x in sq]

    return data, max_len

def dataLabel2Str(data: pd.DataFrame, label_col: str, label_dict: dict) -> pd.DataFrame:
    """Create a meaning column for each unique label.

    Args:
        data (pd.DataFrame): data.
        label_col (str): label column.
        label_dict (dict): dictionary for label descriptions.

    Returns:
        pd.DataFrame: new dataframe with added label meaning column (label_col_str).
    """
    data[label_col + "_str"] = data[label_col].apply(lambda x: str(label_dict[x]))
    return data

def storeDf2Pickle(data: pd.DataFrame, path: str):
    """Store the dataframe in a pickle file.

    Args:
        data (pd.DataFrame): Dataframe to be stored.
        path (str): Path to the created pickle file. 
    """
    assert path.endswith(".pkl"), \
        f"Given file path must end with .pkl, got {path}."
    store_file = open(path, "wb")
    pickle.dump(data, store_file)
    store_file.close()

def openDfFromPickle(path: str) -> pd.DataFrame:
    """Open stored dataframes from pickle files.

    Args:
        path (str): File path.

    Returns:
        pd.DataFrame: Dataframe read.
    """
    assert path.endswith(".pkl"), \
        f"Given file path must end with .pkl, got {path}."

    file = open(path, "rb")
    data = pickle.load(file)
    file.close()
    return data

def bert_tokenize_data(data: pd.DataFrame, tokenizer, keys: List[str], max_len: int = 512, preprocess: bool = False) -> pd.DataFrame:
    if preprocess:
        data = process_df_texts(data, keys)
    
    if tokenizer == None:
        tokenizer = transformers.BertTokenizer.from_pretrained("bert-base-cased")
    
    for key in keys:
        tqdm.pandas(desc=f"Applying bert-tokenization on {key}.")
        data["tmp"] = data[key].progress_apply(lambda x: tokenizer(x, padding='max_length', max_length = max_len, truncation=True, return_tensors="np"))
        data[key + "_bert_input_ids"] = data["tmp"].apply(lambda x: x["input_ids"][0]) 
        data[key + "_bert_token_type_ids"] = data["tmp"].apply(lambda x: x["token_type_ids"][0])
        data[key + "_bert_attention_mask"] = data["tmp"].apply(lambda x: x["attention_mask"][0])
        data = data.drop("tmp", axis=1)
    
    return data, tokenizer



if __name__ == "__main__":
    data_path = "/Users/emrebaloglu/Documents/RL/basic_reinforcement_learning/NLP_datasets/RT_Polarity"
    data = pd.read_csv(data_path + "/rt-polarity-train.csv")
    data = dataLabel2Str(data, "label", {0: "bad", 1: "good"})
    print(data.columns)
    data_val = pd.read_csv(data_path + "/rt-polarity-val.csv")
    data_val = dataLabel2Str(data_val, "label", {0: "bad", 1: "good"})
    data_test = pd.read_csv(data_path + "/rt-polarity-test.csv")
    data_test = dataLabel2Str(data_test, "label", {0: "bad", 1: "good"})
    ########## process with default tokenizer ######################
    """
    data, tokenizer = tokenize_data(data, ["review"], preprocess=True)
    print("n_words_1: ", len(tokenizer.word_index) + 1)
    data, max_len = pad_tokenized_data(data, ["review_tokenized"])
    print("max_len_1: ", max_len)

    
    data_val, tokenizer = tokenize_data(data_val, ["review"], preprocess=True, tokenizer=tokenizer)
    print("n_words_2: ", len(tokenizer.word_index) + 1)
    data_val, _ = pad_tokenized_data(data_val, ["review_tokenized"], max_len=max_len)

    
    data_test, tokenizer = tokenize_data(data_test, ["review"], preprocess=True, tokenizer=tokenizer)
    print("n_words_2: ", len(tokenizer.word_index) + 1)
    data_test, _ = pad_tokenized_data(data_test, ["review_tokenized"], max_len=max_len)
    print(np.stack(data_test["review_tokenized"].values).shape) 
   
    # save the processed data in pickle files

    storeDf2Pickle(data, data_path + "/rt-polarity-train.pkl")
    storeDf2Pickle(data_val, data_path + "/rt-polarity-val.pkl")
    storeDf2Pickle(data_test, data_path + "/rt-polarity-test.pkl")

    data_info = {"path": data_path, "max_len": max_len, "vocab_size": len(tokenizer.word_index) + 1}
    with open(data_path + "/data_info.json", "w") as out:
        json.dump(data_info, out)
    
    data_test = openDfFromPickle(data_path + "/rt-polarity-test.pkl")
    print(data_test.sample(5))
    samples = np.stack(data_test["review_tokenized"].values)
    print(samples.shape)
    ss = []
    for j in range(samples.shape[0]):
        ss.append(np.split(samples[j], 2))
    
    print(len(ss), len(ss[0]))
    print(ss[0][0].shape)"""
    ############################################################################
    
    ################# process with pretrained bert tokenizer ###################
    data = process_df_texts(data, ["review"])
    data_val = process_df_texts(data_val, ["review"])
    data_test = process_df_texts(data_test, ["review"])
    max_len = max(len(x.split(" "))-1 for x in data["review"].values)
    print(max_len)
    data, tokenizer = bert_tokenize_data(data, None, ["review"], max_len=max_len)
    data_val, tokenizer = bert_tokenize_data(data_val, tokenizer, ["review"], max_len=max_len)
    data_test, tokenizer = bert_tokenize_data(data_test, tokenizer, ["review"], max_len=max_len)

    data_info_bert = {"path": data_path, "max_len": max_len, "vocab_size": tokenizer.vocab_size}
    with open(data_path + "/data_info_bert.json", "w") as out:
        json.dump(data_info_bert, out)

    print(data.columns)

    storeDf2Pickle(data, data_path + "/rt-polarity-train-bert.pkl")
    storeDf2Pickle(data_val, data_path + "/rt-polarity-val-bert.pkl")
    storeDf2Pickle(data_test, data_path + "/rt-polarity-test-bert.pkl")

    data = openDfFromPickle(data_path + "/rt-polarity-train-bert.pkl")
    print(data.columns)
    # print(data.head())
    # dd = data.iloc[0,:]
    # print(dd)
    # print(data["review_bert_input_ids"].iloc[0])
    # print(tokenizer.decode(data["review_bert_input_ids"].iloc[0]))

