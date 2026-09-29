import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

ALPHABET = list('ARNDCQEGHILKMFPSTWYV')
IDXTOAA = {i: ALPHABET[i] for i in range(20)}
AATOIDX = {v: k for k, v in IDXTOAA.items()}

def seq_to_one_hot(seq):
    return F.one_hot(torch.tensor([AATOIDX[aa] for aa in list(seq)]), 20)

class Encoder(object):
    """convert between strings and their one-hot representations"""
    def __init__(self, alphabet: str = 'ARNDCQEGHILKMFPSTWYV'):
        self.alphabet = alphabet
        self.a_to_t = {a: i for i, a in enumerate(self.alphabet)}
        self.t_to_a = {i: a for i, a in enumerate(self.alphabet)}

    @property
    def vocab_size(self) -> int:
        return len(self.alphabet)
    
    @property
    def vocab(self) -> np.ndarray:
        return np.array(list(self.alphabet))
    
    @property
    def tokenized_vocab(self) -> np.ndarray:
        return np.array([self.a_to_t[a] for a in self.alphabet])

    def onehotize(self, batch):
        #create a tensor, and then onehotize using scatter_
        onehot = torch.zeros(len(batch), self.vocab_size)
        onehot.scatter_(1, batch.unsqueeze(1), 1)
        return onehot
    
    def encode(self, seq_or_batch, return_tensor=True):
        if isinstance(seq_or_batch, str):
            encoded_list = [self.a_to_t[a] for a in seq_or_batch]
        else:
            encoded_list = [[self.a_to_t[a] for a in seq] for seq in seq_or_batch]
        return torch.tensor(encoded_list) if return_tensor else encoded_list
    
    def decode(self, x):
        if isinstance(x, np.ndarray):
            x = x.tolist()
        elif isinstance(x, torch.Tensor):
            x = x.tolist()

        if isinstance(x[0], list):
            return [''.join([self.t_to_a[t] for t in xi]) for xi in x]
        else:
            return ''.join([self.t_to_a[t] for t in x])


class LengthMaxPool1D(nn.Module):
    def __init__(self, in_dim, out_dim, linear=False, activation='relu'):
        super().__init__()
        self.linear = linear
        if self.linear:
            self.layer = nn.Linear(in_dim, out_dim)

        if activation == 'swish':
            self.act_fn = lambda x: x * torch.sigmoid(100.0*x)
        elif activation == 'softplus':
            self.act_fn = nn.Softplus()
        elif activation == 'sigmoid':
            self.act_fn = nn.Sigmoid()
        elif activation == 'leakyrelu':
            self.act_fn = nn.LeakyReLU()
        elif activation == 'relu':
            self.act_fn = lambda x: F.relu(x)
        else:
            raise NotImplementedError

    def forward(self, x):
        if self.linear:
            x = self.act_fn(self.layer(x))
        x = torch.max(x, dim=1)[0]
        return x


class BaseCNN(nn.Module):
    def __init__(
            self,
            n_tokens: int = 20,
            kernel_size: int = 5 ,
            input_size: int = 256,
            dropout: float = 0.0,
            make_one_hot=True,
            activation: str = 'relu',
            linear: bool=True,
            **kwargs):
        super(BaseCNN, self).__init__()
        self.encoder = nn.Conv1d(n_tokens, input_size, kernel_size=kernel_size)
        self.embedding = LengthMaxPool1D(
            linear=linear,
            in_dim=input_size,
            out_dim=input_size*2,
            activation=activation,
        )
        self.decoder = nn.Linear(input_size*2, 1)
        self.n_tokens = n_tokens
        self.dropout = nn.Dropout(dropout) # TODO: actually add this to model
        self.input_size = input_size
        self._make_one_hot = make_one_hot

    def forward(self, x):
        #onehotize
        if self._make_one_hot:
            x = F.one_hot(x.long(), num_classes=self.n_tokens)
        x = x.permute(0, 2, 1).float()
        # encoder
        x = self.encoder(x).permute(0, 2, 1)
        x = self.dropout(x)
        # embed
        x = self.embedding(x)
        # decoder
        output = self.decoder(x).squeeze(1)
        return output