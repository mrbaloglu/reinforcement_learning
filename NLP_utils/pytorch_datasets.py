from cgi import test
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from typing import Union, Tuple
import pickle
import transformers
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import gc

# def splitCustomDataset(dataset: Dataset, split_size: float, random_state: int) -> Tuple[Dataset, Dataset]:



class NumpyDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray, textual_data: bool=False):
        """torch.Dataset subclass for using numpy.ndarray.

        Arguments:
            x -- The attributes of the data. (n_samples, n_features)
            y -- The targets of the data. (n_samples, n_classes)

        Keyword arguments:
            textual_data -- Whether the given data in x is tokenized text (default: {False})

        """
        super().__init__()
        
        if len(x) != len(y):
            raise ValueError("The parameters x and y must have the same number of samples.")
        
        if len(x.shape) == 1:
            x = np.expand_dims(x, -1)
        if len(y.shape) == 1:
            y = np.expand_dims(y, -1)
        
        
        self.x = torch.Tensor(x)
        self.y = torch.Tensor(y)

        if len(y.shape) == 1:
            self.y = torch.unsqueeze(self.y, 1)
        
        
        if textual_data:
            self.__vocab_size = len(np.unique(x))
        else:
            self.__vocab_size = 0

        if len(x.shape) == 1:
            self.__input_dim = 1
        else:
            self.__input_dim = x.shape[1]
        
        if textual_data:
            self.__output_dim = len(np.unique(y))
        else:
            self.__output_dim = y.shape[1]
        
    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        data = self.x[idx]
        target = self.y[idx]
        return data, target
    
    def get_vocab_size(self) -> int:
        return self.__vocab_size
    
    def get_input_dim(self) -> int:
        return self.__input_dim
    
    def get_output_dim(self) -> int:
        return self.__output_dim
    
    def get_x(self) -> np.ndarray:
        return self.x.detach().numpy()
    def get_y(self) -> np.ndarray:
        return self.y.detach().numpy()
    

class PandasTextDataset(Dataset):
    def __init__(self, data: Union[pd.DataFrame, str], feature_cols: list, target_cols: list, tokenize_data: bool = True):
        """
        torch.Dataset class for using pandas.DataFrame, textual input will be tokenized through pretrained bert tokenizer.

        Arguments:
            data -- Dataframe that contains the data. Either read from csv file (str) or from RAM (pd.DataFrame)
            The dataframe must only contain textual values (objects).
            feature_cols -- List of feature columns. (x)
            target_cols -- List of target columns. (y)
        Raises:
            TypeError or ValueError depending on the argument data.
        """
        super().__init__()

        self.tokenizer = transformers.BertTokenizer.from_pretrained('bert-base-cased')
        self.feature_cols = feature_cols
        self.target_cols = target_cols
        
        if type(data) == str:
            data_ = None
            if data.endswith(".csv"):
                data_ = pd.read_csv(data, encoding = "utf-8")
            elif data.endswith(".pkl"):
                tmp_file = open(data, "rb", encoding="utf-8")
                data_ = pickle.load(tmp_file)
                tmp_file.close()
            else:
                raise ValueError("A csv or pkl file must be given as string.")
            
            if tokenize_data:
                for col in feature_cols:
                    tqdm.pandas(desc=f"Applying bert-tokenization on {col}...")
                    data_[col + str("_bert_tkn")] = data_[col].progress_apply(lambda x: self.tokenizer(x, 
                                    padding='max_length', max_length = 50, truncation=True, return_tensors="pt"))
            self.data = data_.copy()
            del data_
            gc.collect()

        elif type(data) == pd.DataFrame:
            data_ = data.copy()
            if tokenize_data:
                for col in feature_cols:
                    tqdm.pandas(desc=f"Applying bert-tokenization on '{col}'...")
                    data_[col + str("_bert_tkn")] = data_[col].progress_apply(lambda x: self.tokenizer(x, 
                                    padding='max_length', max_length = 50, truncation=True, return_tensors="pt"))
            self.data = data_.copy()
            
        else:
            raise TypeError("The argument data must of type str or pandas.DataFrame")
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx) -> Tuple[dict, torch.Tensor]:
        """
        Get data by index. 

        Args:
            idx (_type_): index

        Returns:
            Tuple[dict, torch.Tensor]: data, target where 
            {data = (data["input_ids"]: torch.Tensor, data["token_type_ids"]: torch.Tensor, data["attention_mask"]: torch.Tensor)}, 
            and {target = label of the data}
        """
        features = []
        for col in self.feature_cols:
            features_ = self.data[col +  "_bert_tkn"][idx] #["input_ids"]
            features.append(features_)
        
        target = torch.Tensor(self.data.loc[idx, self.target_cols])
        return features, target
    
    def split_dataset(self, test_size, random_state):
        train_df, test_df = train_test_split(self.data, test_size=test_size, random_state=random_state)
        train_dataset = PandasTextDataset(train_df, self.feature_cols, self.target_cols, tokenize_data=False)
        test_dataset = PandasTextDataset(test_df, self.feature_cols, self.target_cols, tokenize_data=False)
        return train_dataset, test_dataset

if __name__ == "__main__":
    
    dataset = PandasTextDataset("rt-polarity/rt-polarity-processed.csv", ["review"], ["label"])

    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    for i, data in enumerate(loader):
        xxx, yyy = data
        print(xxx, yyy)
        break