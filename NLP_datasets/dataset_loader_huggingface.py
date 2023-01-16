import platform

sep = '/'
if platform.system() == "Windows":
    sep = '\\'
    print("Running on windows...")


import transformers

from datasets import load_dataset
from sklearn.model_selection import train_test_split
import sys
import os
import json
import pandas as pd
from transformers import AutoTokenizer 




SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
print(SCRIPT_DIR)
sys.path.append(os.path.dirname(SCRIPT_DIR))

params = {
    "dataset_name": "rotten_tomatoes",
    "tokenizer": "distilbert-base-uncased",
    "text_field_name": "text",
    "max_len": 512
}
# label dictionary for imdb and rotten tomatoes
label_dict = {
    '0': "neg",
    '1': "pos",
}

"""# label dictionary for dbpedia
label_dict = {
    '0': "Company",
    '1': "EducationalInstitution",
    '2': "Artist",
    '3': "Athlete",
    '4': "OfficeHolder",
    '5': "MeanOfTransportation",
    '6': "Building",
    '7': "NaturalPlace",
    '8': "Village",
    '9': "Animal",
    '10': "Plant",
    '11': "Album",
    '12': "Film",
    '13': "WrittenWork",
}"""

"""# label dictionary for ag_news:
label_dict = {
    '0': "World",
    '1': "Sports",
    '2': "Business",
    '3': "Sci/Tech",
}"""


import NLP_utils.preprocessing as nlp_processing

if __name__ == "__main__":
    os.makedirs(SCRIPT_DIR + sep + params["dataset_name"], exist_ok=True)
    dataset_name = params["dataset_name"]
    dataset = load_dataset(dataset_name)
    tokenizer = AutoTokenizer.from_pretrained(params["tokenizer"])

    train_df = pd.DataFrame({"text": dataset["train"][params["text_field_name"]], "label": dataset["train"]["label"]})
    train_df = nlp_processing.create_label_desc_column(train_df, "label", label_dict)
    
    test_df = pd.DataFrame({"text": dataset["test"][params["text_field_name"]], "label": dataset["test"]["label"]})
    test_df = nlp_processing.create_label_desc_column(test_df, "label", label_dict)

    train_df, val_df = train_test_split(train_df, test_size=0.2, random_state=42)

    data_list = [test_df, train_df, val_df]
    data_split = ["test", "train", "val"]


    for ii in range(len(data_list)):
        df = data_list[ii]
        df = nlp_processing.process_df_texts(df, ["text"])
        df, _ = nlp_processing.auto_tokenize_data(df, tokenizer, ["text"], max_len=params["max_len"])
        nlp_processing.storeDf2Pickle(df, SCRIPT_DIR + sep + params["dataset_name"] + sep + params["dataset_name"] + "_" + data_split[ii] + "_" + params["tokenizer"] + ".pkl")
    
    data_info = {"max_len": tokenizer.model_max_length, "vocab_size": tokenizer.vocab_size}
    with open(SCRIPT_DIR + sep + params["dataset_name"] + sep + params["tokenizer"] + "_data_info.json", "w") as out:
        json.dump(data_info, out)