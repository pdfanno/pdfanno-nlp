import re
import math
import unicodedata
import torch


class Dataset(torch.utils.data.Dataset):

    def __init__(self, texts, tag2id, token2id, is_train=True):
        instances = []
        self.instances = instances

        unk_token_id = token2id["UNK"]
        for text in texts:
            token_ids = []
            tag_ids = []
            for w in text.words:
                token = w.value()
                # token = token.lower()  # 小文字化
                token = re.sub(r"\d", "0", token)  # 数値を0に置換
                token = normalize_text(token)
                token_id = token2id.get(token, unk_token_id)
                if token_id == unk_token_id and is_train:
                    token2id[token] = len(token2id)
                    token_id = len(token2id) - 1
                token_ids.append(token_id)
                if w.tag is None:  # testing
                    tag_ids.append(-1)
                else:  # training, dev
                    tag_ids.append(tag2id[w.tag])

            token_id = torch.tensor(token_ids, dtype=torch.int64)
            tag_id = torch.tensor(tag_ids, dtype=torch.int64)
            instances.append((token_id, tag_id))

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, idx):
        return self.instances[idx]


def normalize_text(text: str):
    # ハイフンを正規化
    text = re.sub(r"[\u00AD\u02D7\u2010\u2011\u2012\u2013\u2014\u2015\u2043\u2212\u2796\u2E3A\u2E3B\u30FC\uFE58\uFE63\uFF0D\uFF70]", "\u002D", text)
    # unicode正規化
    text = unicodedata.normalize('NFKC', text)
    return text


def encode_positions(positions: list[int], dim: int):
    """
    Positional encoding
    :param positions: [*]
    :param dim: number of output dimensions (int)
    :return: [dim, *]
    """
    p = torch.tensor(positions, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, dim, 2, dtype=torch.float) * -(math.log(10000.0) / dim))
    # pe = torch.zeros(*position.size(), dim)
    pe = torch.zeros(len(positions), dim)
    pe[..., 0::2] = torch.sin(p * div_term)
    pe[..., 1::2] = torch.cos(p * div_term)
    return pe
