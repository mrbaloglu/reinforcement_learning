import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

import numpy as np
from typing import Union, List
# from torchtext.vocab import GloVe
import transformers
import gym

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class Softmax_Policy_Dense_Layers(nn.Module):

    def __init__(self, state_size: int, action_size: int, hidden_layer_dims: List[int]):
        """
        Initialize a policy with softmax probabilities, estimated with a fully-connected neural network.

        Args:
            state_size (int): Size of the states.
            action_size (int): Size of the actions.
            hidden_layer_dims (List[int]): List of units in hidden layers. 
        """
        super(Softmax_Policy_Dense_Layers, self).__init__()

        self.hidden_layers = []
        if len(hidden_layer_dims) > 0:
            self.fc1 = nn.Linear(state_size, hidden_layer_dims[0])
            for ii in range(len(hidden_layer_dims) - 1):
                hidden_layer = nn.Linear(hidden_layer_dims[ii], hidden_layer_dims[ii+1])
                self.hidden_layers.append(hidden_layer)
            out_layer = nn.Linear(hidden_layer_dims[-1], action_size)
            self.hidden_layers.append(out_layer)
        else: 
            self.fc1 = nn.Linear(state_size, action_size)

       
        
    def forward(self, x):
        if len(self.hidden_layers) > 0:
            x = F.relu(self.fc1(x))
            for layer_ix in range(len(self.hidden_layers)-1):
                x = F.relu(self.hidden_layers[layer_ix](x))
            
            x = F.softmax(self.hidden_layers[-1](x), dim=1)
        else:
            x = F.softmax(self.fc1(x), dim=1)
        
        return x
    
    def predict(self, state):
        if type(state) != torch.Tensor:
            state = torch.from_numpy(state).float().unsqueeze(0)
        probs = self.forward(state)#.cpu()
        m = Categorical(probs)
        action = m.sample()
        return action.item(), m.log_prob(action)


class RNN_Baseline_Policy(nn.Module):
    def __init__(self,
                 vocab_size: int,
                 input_dim: int,
                 out_num_classes: int,
                 embed_dim: int = 5,
                 rnn_type: str = "gru",
                 rnn_hidden_size: int = 2,
                 rnn_hidden_out: int = 2,
                 rnn_bidirectional: bool = True,
                 units: int = 50):
        """
        RNN Class for text classification with RL.
        Arguments:
            vocab_size -- specifies the size of the vocabulary to be used in word embeddings.
            input_dim -- specifies the dimension of the features (n_samples, max_sentence_length (input_dim)).
            output_dim -- specifies the number of possible actions for the agent. 

        Keyword Arguments:
            embed_dim -- embed_dim specifies the embedding dimension for the categorical part of the input. (default: {5})
            rnn_type -- specifies the type of the recurrent layer for word embeddings. (default: {"gru"})
            rnn_hidden_size -- specifies the number of stacked recurrent layers. (default: {2})
            rnn_hidden_out -- specifies number of features in the hidden state h of recurrent layer. (default: {2})
            rnn_bidirectional -- specifies whether the recurrent layers be bidirectional. (default: {True})
            units -- specifies the number of neurons in the hidden layers. (default: {50})

        """
        super(RNN_Baseline_Policy, self).__init__()

        self.embed_dim = embed_dim
        self.out_num_classes = out_num_classes
    
        self.embed_enc = nn.Embedding(vocab_size, embed_dim, max_norm=True)
        self.rnn = nn.GRU(embed_dim, rnn_hidden_out, rnn_hidden_size,
                          bidirectional=rnn_bidirectional, batch_first=True)
        if rnn_type == "lstm":
            self.rnn = nn.LSTM(embed_dim, rnn_hidden_out, rnn_hidden_size,
                               bidirectional=rnn_bidirectional, batch_first=True)
        elif rnn_type != "gru":
            raise ValueError("The argument rnn_type must be 'gru' or 'lstm'!")

        rnn_out_dim = rnn_hidden_out * input_dim
        if rnn_bidirectional:
            rnn_out_dim *= 2

        self.flat = nn.Flatten()
        self.fc1 = nn.Linear(rnn_out_dim, units)
        self.fc2 = nn.Linear(units, units)
        self.fc3 = nn.Linear(units, out_num_classes)

    def forward(self, x: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """
        Model prediction function.

        Args:
            x (Union[np.ndarray, torch.Tensor]): data

        Returns:
            torch.Tensor: predictions.
        """

        # type check
        if type(x) != torch.Tensor:
            x = torch.Tensor(x)

        # reshape when there is only one sample
        if len(x.shape) == 1:
            x = torch.unsqueeze(x, 0)
        
        x_c = self.embed_enc(x.int())
        x_c, _ = self.rnn(x_c)
        x_c = self.flat(x_c)

        x = F.relu(self.fc1(x_c))
        x = F.relu(self.fc2(x))
        x = torch.softmax(self.fc3(x), dim=1)

        return x
    
    def predict(self, state):
        if type(state) != torch.Tensor:
            state = torch.from_numpy(state).float().unsqueeze(0)
        probs = self.forward(state)#.cpu()
        m = Categorical(probs)
        action = m.sample()
        return action.item(), m.log_prob(action)


class Transformer_Baseline_Policy(nn.Module):

    def __init__(self,
                 vocab_size: int,
                 input_dim: int,
                 out_num_classes: int,
                 num_heads: int,
                 num_layers: int,
                 pretrained: str = "None",
                 freeze_emb: bool = True,
                 embed_dim: int = 5):
        """
        A classifier template with transformer encoder layers.

        Args:
            vocab_size (int): Vocabulary size of the input.
            input_dim (int): Input dimension.
            out_num_classes (int): Number of classes for the output.
            num_heads (int): The number of heads in the multiheadattention models.
            num_layers (int): The number of stacked transformer-encoder layers to be used.
            pretrained (str, optional): Use pretrained word embeddings or not. Defaults to "None".
            freeze_emb (bool, optional): Freeze the embedding layer or not. Defaults to True.
            embed_dim (int, optional): Dimension of the word embeddings. Defaults to 5.

        Raises:
            NotImplementedError: For pretrained word embeddings.
            ValueError: For incorrectly given argument pretrained
        """
        
        super(Transformer_Baseline_Policy, self).__init__()
        
        self.embed_dim = embed_dim
        self.out_dim = out_num_classes
        self.trns_out_dim = embed_dim * input_dim
        self.embedding = None
        if pretrained == "None":
            self.embedding = nn.Embedding(vocab_size, embed_dim, max_norm=True)
        elif pretrained == "glove":
            """
            glove = GloVe(name='6B', dim=300)
            weights_matrix = np.zeros((vocab_size, 300))
            words_found = 0
            for i, word in enumerate(vocab.get_itos()):
                try: 
                    weights_matrix[i] = glove.get_vecs_by_tokens(word)
                    words_found += 1
                except KeyError:
                    weights_matrix[i] = glove.get_vecs_by_tokens("<unk>")

            self.embedding, num_embeddings, embedding_dim = create_emb_layer(weights_matrix, freeze_emb)
            """
            raise NotImplementedError()
        else:
            raise ValueError("The argument pretrained must be either 'None' or 'glove'.")
        # self.embed_enc = nn.Embedding(vocab_size, embed_dim, max_norm=True)
        
        # print(num_embeddings, embedding_dim)
        trans_enc_layer = nn.TransformerEncoderLayer(d_model = embed_dim, nhead = num_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(trans_enc_layer, num_layers=num_layers)

        self.fc1 = nn.Linear(self.trns_out_dim, out_num_classes)

        self.flatten = nn.Flatten()

    def forward(self, x: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """
        Model prediction function.

        Args:
            x (Union[np.ndarray, torch.Tensor]): data

        Returns:
            torch.Tensor: predictions
        """

        # type check
        if type(x) != torch.Tensor:
            x = torch.Tensor(x)
        
        # reshape when there is only one sample
        if len(x.shape) == 1:
            x = torch.unsqueeze(x, 0)
        
        x = self.embedding(x.int())
        #print(x.shape)
        x = self.transformer(x)
        #print(x.shape)
        #print(cn, hn.shape, 2)
        x = self.flatten(x)
        x = torch.softmax(self.fc1(x), dim=1)

        return x
    
    def predict(self, state):
        if type(state) != torch.Tensor:
            state = torch.from_numpy(state).float().unsqueeze(0)
        probs = self.forward(state)#.cpu()
        m = Categorical(probs)
        action = m.sample()
        return action.item(), m.log_prob(action)

    def embout(self, x):
        return self.embedding(x)

    # freeze the layers until given index
    def freeze_layers(self, index):
        cnt = 0
        for p in self.parameters():
            p.requires_grad = False
            cnt += 1
            if(cnt == index):
              break

class BERT_Baseline_Policy(nn.Module):

    def __init__(self, output_dim, dropout: float = 0.5):

        super(BERT_Baseline_Policy, self).__init__()
        
        self.bert = transformers.BertModel.from_pretrained('bert-base-cased')
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(768, output_dim)

    def forward(self, input_id, mask):
        
        _, pooled_output = self.bert(input_ids=input_id, attention_mask=mask,return_dict=False)
        dropout_output = self.dropout(pooled_output)
        linear_output = self.linear(dropout_output)
        final_layer = torch.softmax(linear_output)

        return final_layer
    
    def predict(self, state):
        if type(state) != torch.Tensor:
            state = torch.from_numpy(state).float().unsqueeze(0)
        probs = self.forward(state)#.cpu()
        m = Categorical(probs)
        action = m.sample()
        return action.item(), m.log_prob(action)


class CNN1DExtractor(BaseFeaturesExtractor):
    def __init__(self, 
                observation_space: gym.spaces.Box,
                n_filter_list: List[int],
                kernel_size: int,
                vocab_size: int,
                embed_dim: int = 5,
                features_dim: int = 256):
        """Create a feature extractor model with 1D conv. nets for SB3 algorithms. 

        Args:
            observation_space (gym.spaces.Box): Observation from the environment.
            n_filter_list (List[int]): List for number of filters in conv. layers.
            kernel_size (int): Size of the sliding window for conv. operations. (will be used in all conv. layers)
            vocab_size (int): Number of unique words in the data for word embeddings.
            embed_dim (int): Embedding dimension. Defaults to 5.
            features_dim (int, optional): Dİmension of the extracted features. Defaults to 256.
        """
        super(CNN1DExtractor, self).__init__(observation_space, features_dim)
        # We assume 1D inputs (n_feature_dim, )

        self.conv_layers = []
        self.kernel_size = kernel_size
        self.out_dim = features_dim
        self.embed_dim = embed_dim

        self.embedding = nn.Embedding(vocab_size, self.embed_dim)
        self.n_filter_list = [self.embed_dim]
        self.n_filter_list.extend(n_filter_list)
        for ii in range(len(self.n_filter_list) - 1):
            self.conv_layers.append(nn.Conv1d(self.n_filter_list[ii], self.n_filter_list[ii+1], 
                                            self.kernel_size, stride=1, padding="same"))

        
        self.flatten = nn.Flatten()

        self.conv_out_dim = 1
        with torch.no_grad():
            #TODO hardcoded unsquueze here, only generic for 1d text inputs 
            
            dummy_data = torch.arange(1, observation_space.shape[0]+1).int().unsqueeze(0)
            dummy_data = self.embedding(dummy_data)
            dummy_data = torch.swapaxes(dummy_data, 1, 2)
            print(dummy_data.shape)
            for layer in self.conv_layers:
                dummy_data = layer(dummy_data)
                print(dummy_data.shape)
            
            dummy_data = self.flatten(dummy_data)
            self.conv_out_dim = dummy_data.shape[1]
            

        self.fc1 = nn.Linear(self.conv_out_dim, self.out_dim)
    

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        x = self.embedding(observations.int())
        x = torch.swapaxes(x, 1, 2)
        # x = torch.unsqueeze(observations, 1) # reshape observations to (n_samples, 1, n_features) for convolution layer
        # print(x.shape)
        for layer in self.conv_layers:
            x = F.relu(layer(x))

        x = self.flatten(x)
        x = F.relu(self.fc1(x))

        return x
       

    # override apply method for conv. layer list (needed in model.to(device))
    def _apply(self, fn):
        super()._apply(fn)
        for layer in self.conv_layers:
            layer = layer._apply(fn)

        return self 


class RNNExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box,
                 vocab_size: int,
                 embed_dim: int = 5,
                 rnn_type: str = "gru",
                 rnn_hidden_size: int = 2,
                 rnn_hidden_out: int = 2,
                 rnn_bidirectional: bool = True,
                 features_dim: int = 256,
                 units: int = 50):
        """RNN Class for text classification with RL.

        Args:
            vocab_size (int): specifies the size of the vocabulary to be used in word embeddings.
            observation_space (gym.spaces.Box):Observation space shaped (n_samples, max_sentence_length (input_dim))
            features_dim (int): specifies the number of possible actions for the agent.
            embed_dim (int, optional): embed_dim specifies the embedding dimension for the categorical part of the input. Defaults to 5.
            rnn_type (str, optional): specifies the type of the recurrent layer for word embeddings. Defaults to "gru".
            rnn_hidden_size (int, optional): specifies the number of stacked recurrent layers. Defaults to 2.
            rnn_hidden_out (int, optional): specifies number of features in the hidden state h of recurrent layer. Defaults to 2.
            rnn_bidirectional (bool, optional): specifies whether the recurrent layers be bidirectional. Defaults to True.
            units (int, optional): specifies the number of neurons in the hidden layers. Defaults to 50.
        """
        super(RNNExtractor, self).__init__(observation_space, features_dim)
        self.embed_dim = embed_dim

        # self.features_dim = features_dim

        input_dim = observation_space.shape[0]
    
        self.embed_enc = nn.Embedding(vocab_size, embed_dim, max_norm=True)
        self.rnn = nn.GRU(embed_dim, rnn_hidden_out, rnn_hidden_size,
                          bidirectional=rnn_bidirectional, batch_first=True)
        if rnn_type == "lstm":
            self.rnn = nn.LSTM(embed_dim, rnn_hidden_out, rnn_hidden_size,
                               bidirectional=rnn_bidirectional, batch_first=True)
        elif rnn_type != "gru":
            raise ValueError("The argument rnn_type must be 'gru' or 'lstm'!")

        rnn_out_dim = rnn_hidden_out * input_dim
        if rnn_bidirectional:
            rnn_out_dim *= 2

        self.flat = nn.Flatten()
        self.fc1 = nn.Linear(rnn_out_dim, units)
        self.fc2 = nn.Linear(units, units)
        self.fc3 = nn.Linear(units, features_dim)
    

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # reshape when there is only one sample
        if len(observations.shape) == 1:
            observations = torch.unsqueeze(observations, 0)
        
        x_c = self.embed_enc(observations.int())
        x_c, _ = self.rnn(x_c)
        x_c = self.flat(x_c)

        x = F.relu(self.fc1(x_c))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))

        return x       


class DummyNN(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box,
                input_dim: int,
                features_dim: int = 256,):
        
        super(DummyNN, self).__init__(observation_space, features_dim)
       
    
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, features_dim)
    

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(observations))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return x       
