"""Moses truecaser trained on ORIGINAL-CASED English (a truecaser trained on lowercased text
cannot restore names such as 'Berlin'). The model itself is trained on lowercased text."""
from __future__ import annotations
from sacremoses import MosesTruecaser, MosesDetokenizer, MosesTokenizer


def train_truecaser(cased_english_lines, path):
    tok = MosesTokenizer(lang="en")
    mtr = MosesTruecaser()
    mtr.train([tok.tokenize(l) for l in cased_english_lines], save_to=path)


def apply_truecaser(path, hyps):
    mtr = MosesTruecaser(path)
    tok, detok = MosesTokenizer(lang="en"), MosesDetokenizer(lang="en")
    out = []
    for h in hyps:
        t = mtr.truecase(tok.tokenize(h, return_str=True, escape=False), return_str=True)
        out.append(detok.detokenize(t.split()))
    return out
