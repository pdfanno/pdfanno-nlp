import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchcrf import CRF


class Tagger(nn.Module):

    def __init__(self, num_tokens=-1, num_tags=None):
        super().__init__()
        embed_dim = 50
        self.embeds = nn.Embedding(num_tokens, embed_dim)
        # self.embeds = nn.Embedding.from_pretrained(embeds, freeze=False)
        # self.conv_char = nn.Conv1d(embed_dim, embed_dim, 3, padding=1)
        # self.lstm = nn.LSTM(input_size=embed_dim, hidden_size=embed_dim, num_layers=1, bidirectional=True)

        h_size = embed_dim
        self.conv1d = nn.Conv1d(h_size, h_size, 3, padding=1)
        self.out = nn.Linear(h_size, num_tags)
        self.crf = CRF(num_tags)

    def forward(self, args, return_feature=False):
        token_id, tag_id = args

        # cs = [self.embeds(c.squeeze(dim=0)) for c in char_ids]  # list[len, embed_dim]
        # c = self.lstm_max_pool(cs)
        # c = self.cnn_select_first(cs)
        x = self.embeds(token_id)
        x = torch.transpose(x, 1, 2)
        for _ in range(5):
            x0 = x
            x = self.conv1d(x)
            x = x * torch.sigmoid(x) + x0
        x = torch.transpose(x, 1, 2)
        # x = x.squeeze(dim=0)

        x = self.out(x)
        x = torch.transpose(x, 0, 1)  # (seq_length, batch, num_tags)
        tag_id = torch.transpose(tag_id, 0, 1)  # (seq_length, batch)

        if self.training:
            loss = self.crf(x, tag_id)
            return -loss
        else:
            z = self.crf.decode(x)
            return torch.tensor(z, dtype=torch.int64)

    def cnn_select_first(self, xs: list[torch.FloatTensor]):
        indexs = []
        cumsum = 0
        for x in xs:
            indexs.append(cumsum)
            cumsum += x.size()[0]
        index = torch.tensor(indexs, dtype=torch.int64)
        x = torch.concat(xs, dim=0)
        x = x.unsqueeze(dim=0)
        x = torch.transpose(x, 1, 2)
        for _ in range(3):
            x0 = x
            x = self.conv_char(x)
            x = x * torch.sigmoid(x) + x0
        # x = self.conv_char(x)
        x = torch.index_select(x, 2, index)
        x = torch.transpose(x, 1, 2)
        return x

    def lstm_max_pool(self, xs: list[torch.FloatTensor]):
        packed = nn.utils.rnn.pack_sequence(xs, enforce_sorted=False)
        y, _ = self.lstm(packed)
        unpacked, _ = nn.utils.rnn.pad_packed_sequence(y, padding_value=-1e10)  # (max_seq_len, batch, dim)
        unpacked = unpacked.permute(1, 2, 0)
        y = nn.functional.adaptive_max_pool1d(unpacked, output_size=1)
        return y.squeeze(dim=2).unsqueeze(dim=0)
