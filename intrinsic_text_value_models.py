import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from typing import List, Union

class DenseStateFeatureExtractor(nn.Module):
    def __init__(self, state_dim: int, output_dim: int, hidden_layer_dims: List[int]):
        
        super().__init__()

        self.hidden_layers = []
        if len(hidden_layer_dims) > 0:
            self.fc1 = nn.Linear(state_dim, hidden_layer_dims[0])
            for ii in range(len(hidden_layer_dims) - 1):
                hidden_layer = nn.Linear(hidden_layer_dims[ii], hidden_layer_dims[ii+1])
                self.hidden_layers.append(hidden_layer)
            out_layer = nn.Linear(hidden_layer_dims[-1], output_dim)
            self.hidden_layers.append(out_layer)
        else: 
            self.fc1 = nn.Linear(state_dim, output_dim)
        
    def forward(self, x):
        if len(self.hidden_layers) > 0:
            x = F.relu(self.fc1(x))
            for layer_ix in range(len(self.hidden_layers)-1):
                x = F.relu(self.hidden_layers[layer_ix](x))
            
            x = self.hidden_layers[-1](x)
        else:
            x = self.fc1(x)
        
        return x


class RNN_Feature_Extractor(nn.Module):
    def __init__(self,
                 vocab_size: int,
                 input_dim: int,
                 output_dim: int,
                 embed_dim: int = 5,
                 rnn_type: str = "gru",
                 rnn_hidden_size: int = 2,
                 rnn_hidden_out: int = 2,
                 rnn_bidirectional: bool = True,
                 units: int = 50):
        """
        RNN Class for text state feature extraction for intrinsic value generation.
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
        super().__init__()

        self.embed_dim = embed_dim
        self.output_dim = output_dim
    
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
        self.fc3 = nn.Linear(units, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
        x = self.fc3(x)

        return x

    
class NextStatePredictor(nn.Module):
    def __init__(self, state_dim: int, hidden_layer_dims: List[int]):
        super().__init__()
        self.hidden_layers = []
        if len(hidden_layer_dims) > 0:
            self.fc1 = nn.Linear(state_dim, hidden_layer_dims[0])
            for ii in range(len(hidden_layer_dims) - 1):
                hidden_layer = nn.Linear(hidden_layer_dims[ii], hidden_layer_dims[ii+1])
                self.hidden_layers.append(hidden_layer)
            out_layer = nn.Linear(hidden_layer_dims[-1], state_dim)
            self.hidden_layers.append(out_layer)
        else: 
            self.fc1 = nn.Linear(state_dim, state_dim)
        
    def forward(self, x):
        if len(self.hidden_layers) > 0:
            x = F.relu(self.fc1(x))
            for layer_ix in range(len(self.hidden_layers)-1):
                x = F.relu(self.hidden_layers[layer_ix](x))
            
            x = self.hidden_layers[-1](x)
        else:
            x = self.fc1(x)
        
        return x

class NextActionPredictor(nn.Module):
    def __init__(self, state_dim: int, hidden_layer_dims: List[int], output_dim: int):
        super().__init__()
        self.hidden_layers = []
        if len(hidden_layer_dims) > 0:
            self.fc1 = nn.Linear(state_dim*2, hidden_layer_dims[0])
            for ii in range(len(hidden_layer_dims) - 1):
                hidden_layer = nn.Linear(hidden_layer_dims[ii], hidden_layer_dims[ii+1])
                self.hidden_layers.append(hidden_layer)
            out_layer = nn.Linear(hidden_layer_dims[-1], state_dim)
            self.hidden_layers.append(out_layer)
        else: 
            self.fc1 = nn.Linear(state_dim*2, output_dim)
        
    def forward(self, x):
        if len(self.hidden_layers) > 0:
            x = F.relu(self.fc1(x))
            for layer_ix in range(len(self.hidden_layers)-1):
                x = F.relu(self.hidden_layers[layer_ix](x))
            
            x = F.softmax(self.hidden_layers[-1](x))
        else:
            x = F.softmax(self.fc1(x), dim=1)
        
        return x
    
    def predict(self, state):
        probs = self.forward(state)#.cpu()
        m = Categorical(probs)
        action = m.sample()
        return action.item(), m.log_prob(action)

