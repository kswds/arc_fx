"""G-loss-S (structured: q = E[e d | x], g = 2q - d^2), G-loss-D (direct Huber+BCE gain), isotonic calibration, placebo threshold.

Training data: records of blocks 3-4 (real and placebo candidates, 1:1, placebo status not an input).
Calibration: isotonic map raw estimate -> realised Huber+BCE gain D, fitted on block 5 records (real + placebo).
Threshold c0: 95th percentile of the calibrated estimate over placebo candidates of blocks 3-5 (fixed before C).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from sklearn.isotonic import IsotonicRegression

from . import cell_reliability as CR
from .message_linear import candidate_mask

RIDGE = (1e1, 1e2, 1e3, 1e4)
MLP = ((8, 1e-3), (8, 1e-2), (16, 1e-3), (16, 1e-2))


def candidate_features(tl, fake=False):
    """(n,10,11,F) selector inputs and the matching d, target arrays."""
    sfx = "_fake" if fake else ""
    d, part = tl["d" + sfx], tl["part" + sfx]
    sd = tl["sd_ref"]
    rel = CR.features(tl["e"], d, tl["reg"], tl["R"], sd)
    n = len(d); R = tl["R"]
    dn = d / sd
    others = d.sum(-1, keepdims=True) - d
    oh = np.eye(R)[tl["reg"]][:, None, None, :].repeat(10, 1).repeat(11, 2)
    cols = [rel["rho_cell"], rel["rho_pair"], np.log1p(rel["n_cell"]), np.clip(rel["se_cell"], 0, 10),
            np.abs(dn), dn] + [part[..., c] / sd for c in range(part.shape[-1])]
    cols += [oh[..., k] for k in range(R)] + [oh[..., k] * np.abs(dn) for k in range(R)]
    rx = lambda v: np.repeat(v[:, :, None], 11, 2)
    cols += [rx(tl["logvol"]), rx(tl["a1"]), rx(tl["a5"]), rx(np.abs(tl["b"])),
             np.sign(d) * np.sign(others), np.abs(others) / sd]
    X = np.clip(np.nan_to_num(np.stack(cols, -1)), -20, 20)
    return X, rel


def flat(tl, sel_rows, fake):
    """Flatten (date, i, j) candidates of timeline positions sel_rows."""
    X, rel = tl["X_fake" if fake else "X"], tl["rel_fake" if fake else "rel"]
    sfx = "_fake" if fake else ""
    cm = candidate_mask()
    take = lambda A: A[sel_rows][:, cm]                               # (n, 100, ...)
    return dict(X=take(X), d=take(tl["d" + sfx]), ed=take(tl["ed" + sfx]), D=take(tl["D" + sfx]),
                pair=np.tile(np.arange(100), (len(sel_rows), 1)), reg=np.repeat(tl["reg"][sel_rows][:, None], 100, 1),
                date=np.repeat(np.asarray(sel_rows)[:, None], 100, 1))


def stack(*ds):
    return {k: np.concatenate([x[k].reshape((-1,) + x[k].shape[2:]) for x in ds]) for k in ds[0]}


class Scaler:
    def __init__(self, X):
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-8

    def __call__(self, X):
        return (X - self.mu) / self.sd


def _design(X, d, kind):
    one = np.ones((len(X), 1))
    Xt = np.concatenate([one, X], 1)
    if kind == "S":
        return np.concatenate([d[:, None] * Xt, (d ** 2)[:, None] * Xt], 1)
    return np.concatenate([Xt, d[:, None] * Xt, (d ** 2)[:, None] * Xt], 1)


class RidgeSel:
    def __init__(self, kind, alpha):
        self.kind, self.alpha, self.name = kind, alpha, f"{kind}_ridge_a{alpha:g}"

    def fit(self, data):
        self.sc = Scaler(data["X"])
        Z = _design(self.sc(data["X"]), data["d"], self.kind)
        self.zs = np.sqrt((Z ** 2).mean(0)) + 1e-12
        Z = Z / self.zs
        y = data["ed"] if self.kind == "S" else data["D"]
        self.ys = y.std() + 1e-12
        self.w = np.linalg.solve(Z.T @ Z + self.alpha * np.eye(Z.shape[1]), Z.T @ (y / self.ys))
        return self

    def q(self, X, d):
        return (_design(self.sc(X), d, self.kind) / self.zs) @ self.w * self.ys

    def raw_gain(self, X, d, cross=0.):
        """S: 2 (q - d * cross) - d^2, cross = sum of already-selected messages (sequential aux); D: direct estimate."""
        q = self.q(X, d)
        return 2 * (q - d * cross) - d ** 2 if self.kind == "S" else q


class MLPSel(nn.Module):
    def __init__(self, kind, hidden, wd, n_in, R, seed):
        super().__init__()
        torch.manual_seed(seed)
        self.kind, self.wd, self.name = kind, wd, f"{kind}_mlp_h{hidden}_wd{wd:g}"
        self.emb = nn.Embedding(100 * R, 4)
        self.R = R
        self.net = nn.Sequential(nn.Linear(n_in + 4, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, 3))
        nn.init.zeros_(self.net[-1].weight); nn.init.zeros_(self.net[-1].bias)

    def forward(self, X, d, cell):
        h = self.net(torch.cat([X, self.emb(cell)], -1))
        if self.kind == "S":
            return d * h[:, 0] + d ** 2 * h[:, 1]
        return h[:, 2] + d * h[:, 0] + d ** 2 * h[:, 1]

    def fit(self, data, epochs=30, patience=5, batch=4096, lr=1e-3):
        self.sc = Scaler(data["X"])
        X = torch.as_tensor(self.sc(data["X"])).float(); d = torch.as_tensor(data["d"]).float()
        self.dscale = float(d.std()) + 1e-12
        d = d / self.dscale
        y = torch.as_tensor(data["ed"] if self.kind == "S" else data["D"]).float()
        self.ys = float(y.std()) + 1e-12; y = y / self.ys
        cell = torch.as_tensor(data["pair"] * self.R + data["reg"]).long()
        dates = np.unique(data["date"]); cut = dates[int(len(dates) * .8)]
        tr = torch.as_tensor(data["date"] < cut); va = ~tr
        opt = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=self.wd)
        g = torch.Generator().manual_seed(0)
        best, state, stale = np.inf, None, 0
        itr = torch.where(tr)[0]
        for ep in range(epochs):
            self.train()
            perm = itr[torch.randperm(len(itr), generator=g)]
            for k in range(0, len(perm), batch):
                ix = perm[k:k + batch]
                loss = ((self(X[ix], d[ix], cell[ix]) - y[ix]) ** 2).mean()
                opt.zero_grad(); loss.backward(); opt.step()
            self.eval()
            with torch.no_grad():
                vl = float(((self(X[va], d[va], cell[va]) - y[va]) ** 2).mean())
            if vl < best - 1e-7:
                best, state, stale = vl, {k: v.clone() for k, v in self.state_dict().items()}, 0
            else:
                stale += 1
                if stale >= patience:
                    break
        self.load_state_dict(state)
        return self

    def raw_gain(self, X, d, cross=0., cell=None):
        q = self._q_units(X, d, cell)
        return 2 * (q - d * cross) - d ** 2 if self.kind == "S" else q

    def _q_units(self, X, d, cell):
        # forward works on d / dscale; h0 * d' + h1 * d'^2 in units of ys -> rescale to e*d units
        self.eval()
        with torch.no_grad():
            Xs = torch.as_tensor(self.sc(X)).float(); ds = torch.as_tensor(d / self.dscale).float(); c = torch.as_tensor(cell).long()
            return self(Xs, ds, c).numpy() * self.ys


class Calibrated:
    """isotonic(raw estimate -> realised D) fitted on block-5 records; threshold c0 from placebo candidates."""

    def __init__(self, sel):
        self.sel = sel

    def raw(self, data, cross=0.):
        X = data["X"].reshape(-1, data["X"].shape[-1]); d = data["d"].ravel()
        cross = np.broadcast_to(cross, np.shape(data["d"])).ravel()
        if isinstance(self.sel, MLPSel):
            cell = (data["pair"] * self.sel.R + data["reg"]).ravel()
            return self.sel.raw_gain(X, d, cross, cell)
        return self.sel.raw_gain(X, d, cross)

    def fit_iso(self, cal):
        raw = self.raw(cal)
        self.iso = IsotonicRegression(out_of_bounds="clip").fit(raw, cal["D"].ravel())
        self.rs = raw.std() + 1e-12
        return self

    def __call__(self, data, cross=0.):
        """isotonic output + a tie-breaker (1e-7 * tanh(raw / sd)) so that plateaus are ordered by the raw estimate."""
        raw = self.raw(data, cross).ravel()
        return (self.iso.predict(raw) + 1e-7 * np.tanh(raw / self.rs)).reshape(np.shape(data["d"]))

    def set_threshold(self, fake_records):
        self.c0 = float(np.quantile(self(fake_records), .95))
        return self
