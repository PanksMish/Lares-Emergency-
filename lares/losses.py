"""L_total = L_MT + beta * L_CLS + lambda * L_latency   (Eq. 14)."""
from __future__ import annotations
import torch
import torch.nn.functional as F
from .data import PAD


def mt_loss(logits, tgt_out, smoothing=0.1):
    return F.cross_entropy(logits.reshape(-1, logits.size(-1)), tgt_out.reshape(-1),
                           ignore_index=PAD, label_smoothing=smoothing)


def latency_loss(g_hat, src, tgt_out):
    """L_latency = 1/m sum_t ( g_hat(t) - (t-1)/r ),  r = m/n   (Eq. 9), averaged over the batch.
    Divided by n so the scale is comparable across lengths (documented deviation: see README)."""
    n = (src != PAD).sum(1).float()
    valid = (tgt_out != PAD)
    m = valid.sum(1).float()
    t = torch.arange(g_hat.size(1), device=g_hat.device)[None, :].float()
    ref = t * (n / m)[:, None]                               # (t-1)/r with 0-based t
    per = ((g_hat - 1 - ref) * valid).sum(1) / m             # g_hat is 1-based
    return (per / n).mean()


def total_loss(out, batch, cfg, g_hat_enabled=True):
    parts = {"mt": mt_loss(out["logits"], batch["tgt_out"], cfg.get("label_smoothing", 0.1))}
    loss = parts["mt"]
    if cfg.get("joint_cls", True):
        parts["cls"] = F.cross_entropy(out["cls_logits"], batch["label"])
        loss = loss + cfg["beta"] * parts["cls"]
    if cfg.get("latency_loss", True) and "g_hat" in out:
        parts["lat"] = latency_loss(out["g_hat"], batch["src"], batch["tgt_out"])
        loss = loss + cfg["lambda"] * parts["lat"]
    parts["total"] = loss
    return loss, parts
