"""Own predictor: b_i,t = bias_i + sum_c direct_c(i,t) + global(i,t), in volatility units (z = return / past vol).

No message from any other currency node enters. Channel encoders are shared across currencies (currency embedding),
separate across channels, and each channel also has a linear path on its explicit lag/flag features.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .data import CHANNELS


def mlp(i, h, o, p=0.):
    return nn.Sequential(nn.Linear(i, h), nn.GELU(), nn.Dropout(p), nn.Linear(h, o))


class OwnModel(nn.Module):
    def __init__(self, dims, explicit, n_global, hidden=32, emb=8, dropout=.1):
        super().__init__()
        self.emb = nn.Embedding(11, emb)
        self.explicit = {c: torch.as_tensor(explicit[c]) for c in CHANNELS}
        self.direct = nn.ModuleDict({c: mlp(dims[c] + emb, hidden, 1, dropout) for c in CHANNELS})
        self.linear = nn.ModuleDict({c: nn.Linear(len(explicit[c]), 1, bias=False) for c in CHANNELS})
        self.glob = mlp(n_global + emb, hidden, 1, dropout)
        self.bias = nn.Parameter(torch.zeros(11))
        for m in list(self.direct.values()) + [self.glob]:
            nn.init.zeros_(m[-1].weight); nn.init.zeros_(m[-1].bias)
        for m in self.linear.values():
            nn.init.zeros_(m.weight)

    def forward(self, X, G):
        """X[c]: (B, 11, F_c); G: (B, n_global). Returns z (B, 10) and components (B, 10, 6) = [bias, fx, rate, equity, macro, global]."""
        B = G.shape[0]
        e = self.emb.weight[1:].unsqueeze(0).expand(B, -1, -1)                       # receivers 1..10
        comps = [self.bias[1:].expand(B, -1)]
        for c in CHANNELS:
            x = X[c][:, 1:]
            comps.append(self.direct[c](torch.cat([x, e], -1)).squeeze(-1) + self.linear[c](x[..., self.explicit[c]]).squeeze(-1))
        g = G.unsqueeze(1).expand(-1, 10, -1)
        comps.append(self.glob(torch.cat([g, e], -1)).squeeze(-1))
        comps = torch.stack(comps, -1)
        return comps.sum(-1), comps


# loss variant (set by train.run from env CPGG_LOSS_MODE): "huber_bce" (default, spec) or "arcfx_dir" (ARC_FX directional core:
# mean softplus(-zp*sign(y)) over the active set |z| >= tau, tau = Q0.40 of |z| on A rows; inactive entries contribute 0)
LOSS_MODE = dict(mode="huber_bce", tau=0.0)


def pred_loss(zp, z, y, huber=1., eta=1., temp=1., reduce=True):
    """Huber((pred - target)/vol) + eta * BCE(pred / (temp * vol), target > 0), with pred and target given in vol units."""
    if LOSS_MODE["mode"] == "arcfx_dir":
        l = F.softplus(-zp * torch.sign(y)) * (z.abs() >= LOSS_MODE["tau"]).float()
        return l.mean() if reduce else l
    l = F.huber_loss(zp, z, delta=huber, reduction="none") + eta * F.binary_cross_entropy_with_logits(zp / temp, (y > 0).float(), reduction="none")
    return l.mean() if reduce else l
