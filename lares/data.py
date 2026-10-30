"""Data preparation: split, SentencePiece BPE, proxy labels, prefix augmentation."""
from __future__ import annotations
import json, os, random
from dataclasses import dataclass
import torch
from torch.utils.data import Dataset

from .lexicon import CATEGORIES, label_sentence

PAD, UNK, BOS, EOS = 0, 1, 2, 3


def read_lines(path):
    with open(path, encoding="utf-8") as f:
        return [l.rstrip("\n") for l in f]


def split_indices(n, seed=13, ratios=(0.90, 0.05, 0.05)):
    """Deterministic 90/5/5 split. Returns dict of index lists."""
    idx = list(range(n))
    random.Random(seed).shuffle(idx)
    n_tr, n_va = int(n * ratios[0]), int(n * ratios[1])
    return {"train": sorted(idx[:n_tr]), "valid": sorted(idx[n_tr:n_tr + n_va]), "test": sorted(idx[n_tr + n_va:])}


def train_spm(texts, model_prefix, vocab_size=30000):
    import sentencepiece as spm
    tmp = model_prefix + ".train.txt"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write("\n".join(texts))
    spm.SentencePieceTrainer.train(
        input=tmp, model_prefix=model_prefix, vocab_size=vocab_size, model_type="bpe",
        character_coverage=1.0, pad_id=PAD, unk_id=UNK, bos_id=BOS, eos_id=EOS,
        input_sentence_size=2_000_000, shuffle_input_sentence=True)
    os.remove(tmp)


class Tok:
    def __init__(self, path):
        import sentencepiece as spm
        self.sp = spm.SentencePieceProcessor(model_file=path)
        self.vocab_size = self.sp.get_piece_size()

    def encode(self, s):            # model works on lowercased text
        return self.sp.encode(s.lower())

    def decode(self, ids):
        return self.sp.decode([i for i in ids if i > EOS])


@dataclass
class Example:
    src: list
    tgt: list
    label: int
    idx: int = -1          # line index in the source file


def build_examples(src_lines, tgt_lines, tok, max_len=128):
    ex = []
    for i, (s, t) in enumerate(zip(src_lines, tgt_lines)):
        a, b = tok.encode(s), tok.encode(t)
        if 0 < len(a) <= max_len and 0 < len(b) <= max_len:
            ex.append(Example(a, b, CATEGORIES.index(label_sentence(s.lower())), i))
    return ex


class PrefixMixDataset(Dataset):
    """With prob. alpha return the full pair; otherwise an artificial-prefix pair
    (X_{<=k}, Y_{<=rho(k)}) with rho(k)=floor(k/n*m) and k ~ U[1, n]  (Sec. 2.4)."""

    def __init__(self, examples, alpha=0.70, prefix_aug=True, seed=0):
        self.ex, self.alpha, self.prefix_aug = examples, alpha, prefix_aug
        self.rng = random.Random(seed)

    def __len__(self):
        return len(self.ex)

    def __getitem__(self, i):
        e = self.ex[i]
        if self.prefix_aug and self.rng.random() > self.alpha:
            n, m = len(e.src), len(e.tgt)
            k = self.rng.randint(1, n)
            r = max(1, (k * m) // n)
            return Example(e.src[:k], e.tgt[:r], e.label, e.idx)
        return e


def collate(batch):
    S = max(len(e.src) for e in batch)
    T = max(len(e.tgt) for e in batch) + 1
    src = torch.full((len(batch), S), PAD)
    tin = torch.full((len(batch), T), PAD)
    tout = torch.full((len(batch), T), PAD)
    for i, e in enumerate(batch):
        src[i, :len(e.src)] = torch.tensor(e.src)
        tin[i, :len(e.tgt) + 1] = torch.tensor([BOS] + e.tgt)
        tout[i, :len(e.tgt) + 1] = torch.tensor(e.tgt + [EOS])
    return {"src": src, "tgt_in": tin, "tgt_out": tout,
            "label": torch.tensor([e.label for e in batch])}
