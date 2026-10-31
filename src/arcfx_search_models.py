"""Isolated search models; the released ARC_FX implementation is unchanged."""
import math
import torch
from torch import nn
from models.arc_fx import StableRelationalScoreGraphFX, usd_pin


class ResidualOriginal(StableRelationalScoreGraphFX):
    def _build_graph(self, h, h_global, A_prev=None, A_override=None):
        out = super()._build_graph(h, h_global, A_prev, A_override)
        # Coupled deterministic q in this path: dropout would defeat exact zero-message cancellation.
        msg = out['adj'] @ self.Wv(h)
        q = lambda m: self.rel_head[3](self.rel_head[1](self.rel_head[0](self.rel_norm(h + m)))).squeeze(-1)
        out['rel_score'] = out['graph_gate'] * (q(msg) - q(torch.zeros_like(msg)))
        out['rel_repr'] = self.rel_norm(h + msg) - self.rel_norm(h)
        return out


class SearchModel(nn.Module):
    def __init__(self, dims, currencies, spec, lookback=10):
        super().__init__()
        self.spec = spec
        self.original = spec['encoder'] == 'original'
        self.dual = spec['loss'].startswith('dual')
        self.no_graph = spec.get('no_graph', False)
        h, n = 48 if self.original else 32, len(currencies)
        if self.original:
            cls = ResidualOriginal if spec.get('residual', False) else StableRelationalScoreGraphFX
            self.backbone = cls(n_ccy=n, currency_names=currencies, dims=dims,
                                hidden=h, top_k=3, graph_rank=8, dropout=.25, edge_dropout=.05)
            self._graph_values = None
            self.backbone.Wv.register_forward_hook(self._capture_values)
            if self.dual:
                self.return_head = nn.Linear(h, 1)
        else:
            d = sum(dims.values())
            self.embedding = nn.Parameter(torch.randn(n, h)*.02)
            if spec['encoder'] == 'gru':
                self.encoder = nn.GRU(d, h, batch_first=True)
            else:
                self.encoder = nn.Sequential(nn.Flatten(1), nn.Linear(lookback*d, h), nn.GELU(), nn.Dropout(.25), nn.Linear(h,h))
            self.norm = nn.LayerNorm(h)
            self.q = nn.Linear(h,h,bias=False)
            self.k = nn.Linear(h,h,bias=False)
            self.value = nn.Linear(h, 4 if spec.get('lowdim') else h, bias=False)
            vd = self.value.out_features
            self.anchor = nn.Parameter(torch.randn(n,n)*.02)
            self.direct = nn.Linear(h, 2 if self.dual else 1)
            self.correction = nn.Sequential(nn.Linear(h+vd,h),nn.GELU(),nn.Linear(h,2 if self.dual else 1))
            self.gate = nn.Linear(h,1)
            nn.init.zeros_(self.gate.weight); nn.init.zeros_(self.gate.bias)
        self.last = {}

    def _capture_values(self, module, inputs, output):
        self._graph_values = output

    def forward(self, batch, zero_graph=False):
        if self.original:
            if zero_graph or self.no_graph:
                batch = dict(batch, A_override=batch['x_local'].new_zeros((len(batch['x_local']),11,11)))
            o = self.backbone(batch)
            o['message'] = o['adj'] @ self._graph_values
            ret = usd_pin(self.return_head(o['node_repr']).squeeze(-1)) if self.dual else None
            self.last = o
            return o['rhat'], ret
        x = torch.cat([batch['x_local'],batch['x_rate'],batch['x_equity'],batch['x_countrymacro'],
                       batch['x_global'].unsqueeze(2).expand(-1,-1,11,-1)],-1)
        b,l,n,d = x.shape
        seq = x.permute(0,2,1,3).reshape(b*n,l,d)
        h = self.encoder(seq)
        if isinstance(h,tuple): h = h[1][-1]
        h = self.norm(h.reshape(b,n,-1)+self.embedding)
        e = self.q(h) @ self.k(h).transpose(1,2)/math.sqrt(h.shape[-1])
        if self.spec.get('anchor',False): e = e+self.anchor
        eye = torch.eye(n,device=x.device,dtype=torch.bool).unsqueeze(0)
        mode = self.spec.get('graph','topk')
        if mode == 'soft':
            a = e.sign()*torch.relu(e.abs()-.05)
            a = a.masked_fill(eye,0)
            a = a / a.abs().sum(-1,keepdim=True).clamp_min(1.)
        else:
            mag = e.abs().masked_fill(eye,-torch.inf)
            if mode == 'topk':
                v,ix = mag.topk(3,-1)
                mag = torch.full_like(mag,-torch.inf).scatter(-1,ix,v)
            a = mag.softmax(-1)*e.tanh()
        if zero_graph or self.no_graph: a = torch.zeros_like(a)
        m = a @ self.value(h)
        g = self.gate(h).sigmoid()
        delta = self.correction(torch.cat([h,m],-1))-self.correction(torch.cat([h,torch.zeros_like(m)],-1))
        s = self.direct(h)+g*delta
        self.last = {'adj':a,'edge_logits':e.masked_fill(eye,0),'graph_gate':g,
                     'message':m,'components':{'rel':delta},'regularization':{}}
        return usd_pin(s[:,:,0]), usd_pin(s[:,:,1]) if self.dual else None
