import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack

from modules import GlobalAttention


class EncoderBase(nn.Module):
    """Base encoder class. """
    def _check_args(self, input, lengths=None, hidden=None):
        seq_len, batch_size, n_feats = input.size()
        if lengths is not None:
            batch_size_, = lengths.size()
            assert batch_size == batch_size_, "Batch size mismatch."

    def forward(self, input, lengths=None, hidden=None):
        raise NotImplementedError()


class MeanEncoder(EncoderBase):
    """Encoder that takes average over the inputs. i.e. mean pooling. """
    def __init__(self, n_layers, embeddings):
        super(MeanEncoder, self).__init__()
        self.n_layers = n_layers
        self.embeddings = embeddings

    def forward(self, input, lengths=None, hidden=None):
        self._check_args(input, lengths, hidden)

        emb = self.embeddings(input)
        seq_len, batch_size, embed_size = emb.size()
        mean = emb.mean(0).expand(self.n_layers, batch_size, embed_size)
        return (mean, mean), emb


class RNNEncoder(EncoderBase):
    """Generic recurrent neural network encoder. """
    def __init__(self, rnn_type, bidirectional, n_layers,
                 hidden_size, dropout=0.0, embeddings=None):
        super(RNNEncoder, self).__init__()
        assert embeddings is not None

        n_directions = 2 if bidirectional else 1
        self.embeddings = embeddings
        self.rnn = getattr(nn, rnn_type)(
            input_size=embeddings.embed_size,
            hidden_size=hidden_size,
            num_layers=n_layers,
            dropout=dropout,
            bidirectional=bidirectional)

    def forward(self, input, lengths=None, hidden=None):
        self._check_args(input, lengths, hidden)
        assert lengths is not None, "lengths is required for RNNEncoder."
        
        emb = self.embeddings(input)
        seq_len, batch_size, embed_size = emb.size()

        packed_emb = emb
        lengths = lengths.view(-1).tolist()
        packed_emb = pack(emb, lengths)

        outputs, h_t = self.rnn(packed_emb, hidden)
        outputs = unpack(outputs)[0]

        return h_t, outputs


class RNNDecoderBase(nn.Module):
    """Base recurrent decoder class. """
    def __init__(self, rnn_type, bidirectional_encoder, n_layers,
                 hidden_size, attn_type='general',
                 coverage_attn=False, copy_attn=False,
                 dropout=0.0, embeddings=None,
                 reuse_copy_attn=False):
        super(RNNDecoderBase, self).__init__()

        self.decoder_type = 'rnn'
        self.bidirectional_encoder = bidirectional_encoder
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.embeddings = embeddings
        self.dropout = nn.Dropout(dropout)

        self.rnn = self._build_rnn(rnn_type, self._input_size, hidden_size,
                                   n_layers, dropout)

        self._coverage = coverage_attn
        self.attn = GlobalAttention(hidden_size, coverage=coverage_attn,
                                    attn_type=attn_type)

        self._copy = True if copy_attn else False
        if copy_attn and not reuse_copy_attn:
            self.copy_attn = GlobalAttention(hidden_size, attn_type=attn_type)
        self._reuse_copy_attn = reuse_copy_attn

    def forward(self, input, context, state, context_lengths=None):
        pass
