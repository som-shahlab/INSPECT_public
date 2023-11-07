import torch.nn as nn
from . import vision_backbones


import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class Attention(nn.Module):
    """
    Adapted from:
        https://github.com/GuanshuoXu/RSNA-STR-Pulmonary-Embolism-Detection/blob/main/trainall/2nd_level/seresnext101_192.py
    """

    def __init__(self, feature_dim, step_dim, bias=True, **kwargs):
        super(Attention, self).__init__(**kwargs)
        print("=" * 80)
        print("Using attention")
        print("=" * 80)

        self.supports_masking = True
        self.bias = bias
        self.feature_dim = feature_dim
        self.step_dim = step_dim
        self.features_dim = 0

        weight = torch.zeros(feature_dim, 1)
        nn.init.xavier_uniform_(weight)
        self.weight = nn.Parameter(weight)
        self.weight = self.weight.type(torch.float32)

        if bias:
            self.b = nn.Parameter(torch.zeros(step_dim))
        self.b = self.b.type(torch.float32)

    def forward(self, x, mask=None):
        feature_dim = self.feature_dim
        step_dim = self.step_dim

        eij = torch.mm(x.contiguous().view(-1, feature_dim), self.weight).view(
            -1, step_dim
        )

        if self.bias:
            eij = eij + self.b

        eij = torch.tanh(eij)
        a = torch.exp(eij)

        if mask is not None:
            a = a * mask

        a = a / torch.sum(a, 1, keepdim=True) + 1e-10
        weighted_input = x * torch.unsqueeze(a, -1)

        return torch.sum(weighted_input, 1), self.weight


class RNNSequentialEncoder(nn.Module):
    """Model to encode series of encoded 2D CT slices using RNN

    Args:
        feature_size (int): number of features for input feature vector
        rnn_type (str): either lstm or gru
        hidden_size (int): number of hidden units
        bidirectional (bool): use bidirectional rnn
        num_layers (int): number of rnn layers
        dropout_prob (float): dropout probability
    """

    def __init__(
        self,
        feature_size: int,
        rnn_type: str = "lstm",
        hidden_size: int = 128,
        bidirectional: bool = True,
        num_layers: int = 1,
        dropout_prob: float = 0.0,
    ):
        super(RNNSequentialEncoder, self).__init__()

        self.feature_size = feature_size
        self.rnn_type = rnn_type
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.dropout_prob = dropout_prob
        self.num_layers = num_layers

        if self.rnn_type not in ["LSTM", "GRU"]:
            raise Exception("RNN type has to be either LSTM or GRU")

        self.rnn = getattr(nn, rnn_type)(
            self.feature_size,
            self.hidden_size,
            batch_first=True,
            num_layers=self.num_layers,
            dropout=self.dropout_prob,
            bidirectional=bidirectional,
        )

    def forward(self, x):
        x = x.transpose(0, 1)
        x, _ = self.rnn(x)  # (Slice, Batch, Feature)
        x = x.transpose(0, 1)  # (Batch, Slice, Feature)
        return x


def get_transformer(n_layers, seq_input_size, nhead, hidden_size, dropout_prob):
    layers = torch.nn.Sequential()

    for _ in range(n_layers):
        layers.append(
            nn.TransformerEncoderLayer(
                d_model=seq_input_size,
                nhead=nhead,
                batch_first=True,
                dim_feedforward=hidden_size,
                dropout=dropout_prob,
            )
        )

    return layers


class Model1D(nn.Module):
    def __init__(self, cfg, num_classes=1):
        super(Model1D, self).__init__()

        # rnn input size
        # seq_input_size = cfg.dataset.pretrain_args.model_type
        model_fn = getattr(vision_backbones, cfg.dataset.pretrain_args.model_type)
        model, seq_input_size = model_fn()
        if cfg.trainer.position_encoding:
            seq_input_size += 1
        del model

        if cfg.dataset.contextualize_slice:
            seq_input_size = seq_input_size * 3

        # classifier input size
        cls_input_size = cfg.model.seq_encoder.hidden_size

        if cfg.model.seq_encoder.rnn_type == "transformer":
            cls_input_size = seq_input_size
            self.seq_encoder = get_transformer(
                n_layers=cfg.model.seq_encoder.num_layers,
                seq_input_size=seq_input_size,
                nhead=16,
                hidden_size=cfg.model.seq_encoder.hidden_size,
                dropout_prob=cfg.model.seq_encoder.dropout_prob,
            )
        elif cfg.model.seq_encoder.rnn_type in ["LSTM", "GRU"]:
            if cfg.model.seq_encoder.bidirectional:
                cls_input_size = cls_input_size * 2
            self.seq_encoder = RNNSequentialEncoder(
                seq_input_size, **cfg.model.seq_encoder
            )
        else:
            raise Exception("")

        if "attention" in cfg.model.aggregation:
            self.attention = Attention(cls_input_size, cfg.dataset.num_slices)

        if cfg.model.aggregation == "attention+max":
            cls_input_size = cls_input_size * 2

        # self.batch_norm_layer = torch.nn.BatchNorm1d(cls_input_size)
        self.classifier = nn.Linear(cls_input_size, num_classes)
        self.cfg = cfg

    def forward(self, x, get_features=False, mask=None):
        x = self.seq_encoder(x)
        x, w = self.aggregate(x, mask)
        # x = self.batch_norm_layer(x)
        pred = self.classifier(x)
        return pred, x

    def aggregate(self, x, mask=None):
        if self.cfg.model.aggregation == "attention":
            return self.attention(x, mask)
        elif self.cfg.model.aggregation == "attention+max":
            max_pool, _ = torch.max(x, 1)
            attn_pool, w = self.attention(x, mask)
            x = torch.cat((max_pool, attn_pool), 1)
            return x, w
        elif self.cfg.model.aggregation == "mean":
            x = torch.mean(x, 1)
            return x, None
        elif self.cfg.model.aggregation == "max":
            x, _ = torch.max(x, 1)
            return x, None
        else:
            raise Exception(
                "Aggregation method should be one of 'attention', 'mean' or 'max'"
            )
