"""LARES model: chunk-causal incremental encoder, Transformer decoder, classification head
on the final decoder state, and a training-time soft read/write policy (Sec. 2.3, 3)."""
from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .data import PAD


def sinusoid(max_len, d):
    pe = torch.zeros(max_len, d)
    pos = torch.arange(max_len).unsqueeze(1)
    div = torch.exp(torch.arange(0, d, 2) * (-math.log(10000.0) / d))
    pe[:, 0::2], pe[:, 1::2] = torch.sin(pos * div), torch.cos(pos * div)
    return pe


class LARES(nn.Module):
    def __init__(self, vocab_size, d_model=512, d_ff=1024, heads=8, enc_layers=3, dec_layers=3,
                 dropout=0.1, n_classes=5, chunk_tokens=2, chunk_encoder=True, max_len=512):
        super().__init__()
        self.d, self.chunk, self.chunk_encoder = d_model, chunk_tokens, chunk_encoder
        self.emb = nn.Embedding(vocab_size, d_model, padding_idx=PAD)   # shared src/tgt/output
        nn.init.normal_(self.emb.weight, 0.0, d_model ** -0.5)          # tied output layer needs small init
        with torch.no_grad():
            self.emb.weight[PAD].zero_()
        self.register_buffer("pe", sinusoid(max_len, d_model), persistent=False)
        mk = lambda: dict(d_model=d_model, nhead=heads, dim_feedforward=d_ff, dropout=dropout,
                          batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(**mk()), enc_layers,
                                             norm=nn.LayerNorm(d_model), enable_nested_tensor=False)
        self.decoder = nn.TransformerDecoder(nn.TransformerDecoderLayer(**mk()), dec_layers,
                                             norm=nn.LayerNorm(d_model))
        self.drop = nn.Dropout(dropout)
        self.cls = nn.Linear(d_model, n_classes)                       # Eq. (13)
        # soft policy p_{t,j} = sigmoid(W_a [h_j^enc ; h_t^dec] + b_a), W_a a row vector (Eq. 8)
        self.pol_enc, self.pol_dec = nn.Linear(d_model, 1, bias=False), nn.Linear(d_model, 1)

    # ---- helpers
    def embed(self, ids):
        return self.drop(self.emb(ids) * math.sqrt(self.d) + self.pe[: ids.size(1)])

    def chunk_mask(self, S, device):
        """True = blocked. Token i may attend to tokens in chunks <= chunk(i) (incremental encoder)."""
        c = torch.arange(S, device=device) // self.chunk
        return c[None, :] > c[:, None]

    # ---- forward pieces
    def encode(self, src):
        mask = self.chunk_mask(src.size(1), src.device) if self.chunk_encoder else None
        return self.encoder(self.embed(src), mask=mask, src_key_padding_mask=(src == PAD))

    def decode(self, tgt_in, memory, src):
        T = tgt_in.size(1)
        causal = torch.triu(torch.ones(T, T, dtype=torch.bool, device=tgt_in.device), 1)
        h = self.decoder(self.embed(tgt_in), memory, tgt_mask=causal,
                         tgt_key_padding_mask=(tgt_in == PAD), memory_key_padding_mask=(src == PAD))
        return h, F.linear(h, self.emb.weight)                         # tied output projection

    def final_state(self, h, tgt_in):
        """Decoder state at the last valid target position (h_m^dec)."""
        last = (tgt_in != PAD).sum(1) - 1
        return h[torch.arange(h.size(0)), last]

    def expected_delay(self, memory, h, src):
        """Differentiable expected source delay g_hat(t) from the soft policy (Eq. 8-9).
        alpha_{t,j} = p_{t,j} prod_{k<j} (1-p_{t,k}); the last valid source position gets p=1."""
        pad = (src == PAD)
        logit = self.pol_enc(memory).squeeze(-1)[:, None, :] + self.pol_dec(h).squeeze(-1)[:, :, None]
        p = torch.sigmoid(logit).masked_fill(pad[:, None, :], 1.0)
        last = (~pad).sum(1) - 1
        p[torch.arange(src.size(0)), :, last] = 1.0
        stay = torch.cumprod(1 - p, dim=-1)
        alpha = p * torch.cat([torch.ones_like(stay[..., :1]), stay[..., :-1]], dim=-1)
        pos = torch.arange(1, src.size(1) + 1, device=src.device, dtype=alpha.dtype)
        return (alpha * pos).sum(-1)                                    # (B, T)

    def forward(self, src, tgt_in, want_policy=False):
        mem = self.encode(src)
        h, logits = self.decode(tgt_in, mem, src)
        out = {"logits": logits, "cls_logits": self.cls(self.final_state(h, tgt_in))}
        if want_policy:
            out["g_hat"] = self.expected_delay(mem, h, src)
        return out

    def n_params(self):
        return sum(p.numel() for p in self.parameters())
