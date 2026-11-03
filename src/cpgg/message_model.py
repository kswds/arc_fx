"""Neighbour correction messages d_{i<-j,t} = a_{i<-j,t} * sum_c beta_c(receiver, regime, e_i, e_j)^T v_{j,t,c}.

v_{j,t,c} = [small_encoder_c(X_c[j]), explicit lag features and flags of X_c[j]]  (no edge bias; no global copy in v).
a in (0, 1) is a transmission strength; the sign of the correction lives in delta.
Everything is in volatility units of the receiver; prediction(S) = b + sum_{j in S} d_ij is additive in S.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .data import CHANNELS
from .own_model import mlp


class MessageModel(nn.Module):
    def __init__(self, dims, explicit, n_global, hidden=32, emb=8, enc=8, dropout=.1, use_explicit=True, use_regime=True,
                 pair_table=True):
        super().__init__()
        self.pair_table = pair_table
        self.explicit = {c: torch.as_tensor(explicit[c]) for c in CHANNELS}
        self.use_explicit, self.use_regime = use_explicit, use_regime
        self.emb = nn.Embedding(11, emb)
        self.n_global = n_global
        rin = sum(dims.values()) + n_global + emb
        self.receiver = nn.Sequential(nn.Linear(rin, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, hidden), nn.GELU())
        self.enc = nn.ModuleDict({c: nn.Sequential(nn.Linear(dims[c], hidden), nn.GELU(), nn.Linear(hidden, enc)) for c in CHANNELS})
        self.vdim = {c: enc + (len(explicit[c]) if use_explicit else 0) for c in CHANNELS}
        tin = hidden + (n_global if use_regime else 0) + 2 * emb
        self.trunk = nn.Sequential(nn.Linear(tin, hidden), nn.GELU(), nn.Dropout(dropout))
        self.beta = nn.ModuleDict({c: nn.Linear(hidden, self.vdim[c]) for c in CHANNELS})
        for m in self.beta.values():
            nn.init.normal_(m.weight, std=1e-3); nn.init.zeros_(m.bias)
        # pair-specific linear coefficients (receiver i, sender j): a direct path for edge-specific readout of v
        # (fix attempt 1: without it a pair-specific coefficient can only arise through embedding x trunk interactions)
        if pair_table:
            self.pair = nn.ParameterDict({c: nn.Parameter(torch.zeros(10, 11, self.vdim[c])) for c in CHANNELS})
        self.weight = mlp(hidden + enc * len(CHANNELS) + (n_global if use_regime else 0) + 2 * emb, hidden, 1, dropout)
        nn.init.zeros_(self.weight[-1].weight); nn.init.zeros_(self.weight[-1].bias)

    def forward(self, X, G):
        """Returns d (B,10,11), channel parts (B,10,11,4) = a * delta_c, a (B,10,11), delta (B,10,11). Self-pairs are 0."""
        B = G.shape[0]
        E = self.emb.weight
        r = self.receiver(torch.cat([torch.cat([X[c] for c in CHANNELS], -1)[:, 1:], G.unsqueeze(1).expand(-1, 10, -1),
                                     E[1:].unsqueeze(0).expand(B, -1, -1)], -1))                    # (B,10,H)
        encs = {c: self.enc[c](X[c]) for c in CHANNELS}                                             # (B,11,enc)
        v = {c: torch.cat([encs[c], X[c][..., self.explicit[c]]], -1) if self.use_explicit else encs[c] for c in CHANNELS}
        ri = r.unsqueeze(2).expand(-1, -1, 11, -1)
        ei = E[1:].view(1, 10, 1, -1).expand(B, -1, 11, -1)
        ej = E.view(1, 1, 11, -1).expand(B, 10, -1, -1)
        parts = [ri, ei, ej]
        if self.use_regime:
            parts.append(G.view(B, 1, 1, -1).expand(-1, 10, 11, -1))
        h = self.trunk(torch.cat(parts, -1))                                                        # (B,10,11,H)
        beta = {c: self.beta[c](h) + (self.pair[c] if self.pair_table else 0.) for c in CHANNELS}
        delta_c = torch.stack([(beta[c] * v[c].unsqueeze(1)).sum(-1) for c in CHANNELS], -1)       # (B,10,11,4)
        sj = torch.cat([encs[c] for c in CHANNELS], -1).unsqueeze(1).expand(-1, 10, -1, -1)
        a = torch.sigmoid(self.weight(torch.cat(parts[:1] + [sj] + parts[1:], -1))).squeeze(-1)    # (B,10,11)
        selfmask = 1. - torch.eye(11, device=G.device)[1:].unsqueeze(0)
        a = a * selfmask
        part = a.unsqueeze(-1) * delta_c
        return dict(d=part.sum(-1), part=part, a=a, delta=delta_c.sum(-1) * selfmask)


def candidate_mask(B=None):
    """(10, 11) bool: receiver i (row i = currency i+1) may receive from every currency except itself."""
    m = ~torch.eye(11, dtype=torch.bool)[1:]
    return m if B is None else m.unsqueeze(0).expand(B, -1, -1)
