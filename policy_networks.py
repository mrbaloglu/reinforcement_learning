import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

import numpy as np
from typing import Union
from torchtext.vocab import GloVe
import transformers


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
                 units: int = 50, 
                 device: torch.device = torch.device("cpu")):
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
        self.device = device

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
    
    def act(self, state):
        if type(state) != torch.Tensor:
            state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
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
                 embed_dim: int = 5,
                 device: torch.device = torch.device("cpu")):
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
        self.device = device
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
    
    def act(self, state):
        if type(state) != torch.Tensor:
            state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
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

    def __init__(self, output_dim, dropout: float = 0.5, device: torch.device = torch.device("cpu")):

        super(BERT_Baseline_Policy, self).__init__()
        self.device = device
        self.bert = transformers.BertModel.from_pretrained('bert-base-cased')
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(768, output_dim)

    def forward(self, input_id, mask):
        
        _, pooled_output = self.bert(input_ids=input_id, attention_mask=mask,return_dict=False)
        dropout_output = self.dropout(pooled_output)
        linear_output = self.linear(dropout_output)
        final_layer = torch.softmax(linear_output)

        return final_layer
    
    def act(self, state):
        if type(state) != torch.Tensor:
            state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        probs = self.forward(state)#.cpu()
        m = Categorical(probs)
        action = m.sample()
        return action.item(), m.log_prob(action)

        