"""Gain selectors.

G-loss: regress the realised loss difference D (continuous, negative values kept) on selector features.
G-rel : estimate the reliability rho(x) with  min sum w (e - rho(x) d)^2  (= d^2-weighted regression of e/d on x);
        add a candidate when rho > 1/2. Exact for squared error (D = 2ed - d^2); an approximation for Huber + BCE.
Both: ridge (closed form) or a small MLP; feature standardisation from the training rows only; date-level weights.
"""
from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn as nn

INTER = ["b", "cum", "g", "recv", "send", "log_vol"]       # columns multiplied by d for the ridge G-loss design


class Standard:
    def __init__(self, x, w):
        w = w / w.sum()
        self.mu = (x * w[:, None]).sum(0)
        self.sd = ((x - self.mu) ** 2 * w[:, None]).sum(0).sqrt().clamp_min(1e-6)

    def __call__(self, x):
        return (x - self.mu) / self.sd


def _design_loss(xs, d, dsd):
    dd = (d / dsd).unsqueeze(-1)
    return torch.cat([xs, dd * xs, dd ** 2, torch.ones_like(dd)], -1)


def _design_rel(xs):
    return torch.cat([xs, torch.ones(len(xs), 1)], -1)


class Selector:
    """kind in {'loss', 'rel'}; model in {'ridge', 'mlp'}."""

    def __init__(self, kind, model, params, seed=0):
        self.kind, self.model, self.params, self.seed = kind, model, dict(params), seed

    # -------------------------------------------------------------- fitting
    def fit(self, rec, val_mask=None, mlp_cfg=None):
        x, w, d = rec["x"].double(), rec["w"].double(), rec["d"].double()
        self.st = Standard(x, w)
        xs = self.st(x)
        self.dsd = float(torch.sqrt((d ** 2 * w).sum() / w.sum()).clamp_min(1e-8))
        if self.kind == "loss":
            self.Dsd = float(torch.sqrt((rec["D"].double() ** 2 * w).sum() / w.sum()).clamp_min(1e-12))
        if self.model == "ridge":
            self._fit_ridge(xs, rec, w, d)
        else:
            self._fit_mlp(xs, rec, w, d, val_mask, mlp_cfg)
        return self

    def _fit_ridge(self, xs, rec, w, d):
        if self.kind == "loss":
            Z = _design_loss(xs, d, self.dsd); t = rec["D"].double() / self.Dsd
        else:
            Z = _design_rel(xs) * d.unsqueeze(-1); t = rec["e"].double()
        sw = w.sum()
        A = (Z * w[:, None]).T @ Z / sw                                  # weighted second moments
        pen = 1e-3 * self.params["alpha"] * torch.eye(Z.shape[1], dtype=Z.dtype)   # alpha 1/10/100 -> 0.001/0.01/0.1
        self.coef = torch.linalg.solve(A + pen, (Z * w[:, None]).T @ t / sw)

    def _fit_mlp(self, xs, rec, w, d, val_mask, cfg):
        torch.manual_seed(self.seed)
        h = self.params["hidden"]
        self.net = nn.Sequential(nn.Linear(xs.shape[1] + (1 if self.kind == "loss" else 0), h), nn.GELU(),
                                 nn.Linear(h, h), nn.GELU(), nn.Linear(h, 1)).double()
        if self.kind == "rel":
            nn.init.zeros_(self.net[-1].weight); nn.init.zeros_(self.net[-1].bias)
        opt = torch.optim.AdamW(self.net.parameters(), lr=cfg["lr"], weight_decay=self.params["weight_decay"])
        tr = ~val_mask
        best, state, stale = np.inf, None, 0
        g = torch.Generator().manual_seed(self.seed)
        idx = tr.nonzero().squeeze(-1)
        for ep in range(cfg["epochs"]):
            self.net.train()
            perm = idx[torch.randperm(len(idx), generator=g)]
            for k in range(0, len(perm), cfg["batch_size"]):
                b = perm[k:k + cfg["batch_size"]]
                loss = self._obj(xs[b], rec, b, w[b], d[b])
                opt.zero_grad(); loss.backward(); opt.step()
            self.net.eval()
            with torch.no_grad():
                vb = val_mask.nonzero().squeeze(-1)
                vl = float(self._obj(xs[vb], rec, vb, w[vb], d[vb]))
            if vl < best - 1e-9:
                best, state, stale = vl, copy.deepcopy(self.net.state_dict()), 0
            else:
                stale += 1
                if stale >= cfg["patience"]:
                    break
        self.net.load_state_dict(state)
        self.val_obj, self.epochs = best, ep + 1

    def _obj(self, xs, rec, b, w, d):
        if self.kind == "loss":
            p = self.net(torch.cat([xs, (d / self.dsd).unsqueeze(-1)], -1)).squeeze(-1)
            r = p - rec["D"][b].double() / self.Dsd
        else:
            rho = .5 + self.net(xs).squeeze(-1)
            r = rec["e"][b].double() - rho * d
        return (w * r ** 2).sum() / w.sum()

    # -------------------------------------------------------------- inference
    @torch.no_grad()
    def gain(self, x, d):
        """Decision score in comparable units: G-loss -> estimated D (loss units); G-rel -> (rho - 1/2) * d^2 (>0 iff rho > 1/2)."""
        xs = self.st(x.double()); d = d.double()
        if self.kind == "loss":
            if self.model == "ridge":
                return (_design_loss(xs, d, self.dsd) @ self.coef * self.Dsd).float()
            return (self.net(torch.cat([xs, (d / self.dsd).unsqueeze(-1)], -1)).squeeze(-1) * self.Dsd).float()
        return ((self.rho(x) - .5) * d ** 2).float()

    @torch.no_grad()
    def rho(self, x):
        xs = self.st(x.double())
        if self.model == "ridge":
            return (_design_rel(xs) @ self.coef).float()
        return (.5 + self.net(xs).squeeze(-1)).float()
