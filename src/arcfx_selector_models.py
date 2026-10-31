"""Change only graph selection, keeping each backbone's score and other paths fixed."""
import math
import torch
from arcfx_search_models import SearchModel
from models.arc_fx import StableRelationalScoreGraphFX, topk_softmax, usd_pin


def select_edges(score,mode,tau=0.,k=3):
    n=score.shape[-1]
    eye=torch.eye(n,device=score.device,dtype=torch.bool)
    safe=score.masked_fill(eye,0.)
    if mode=='soft_threshold':
        a=safe.sign()*torch.relu(safe.abs()-tau)
        a=a.masked_fill(eye,0.)
        return a/a.abs().sum(-1,keepdim=True).clamp_min(1.)
    magnitude=safe.abs().masked_fill(eye,-torch.inf)
    weight=topk_softmax(magnitude,k) if mode=='topk' else magnitude.softmax(-1)
    return weight*safe.tanh()


class CompactSelector(SearchModel):
    def __init__(self,dims,currencies,mode,tau=0.):
        super().__init__(dims,currencies,dict(encoder='mlp',loss='bce',graph='dense'))
        self.mode=mode;self.tau=tau

    def forward(self,batch):
        x=torch.cat([batch['x_local'],batch['x_rate'],batch['x_equity'],batch['x_countrymacro'],
                     batch['x_global'].unsqueeze(2).expand(-1,-1,11,-1)],-1)
        b,l,n,d=x.shape
        h=self.norm(self.encoder(x.permute(0,2,1,3).reshape(b*n,l,d)).reshape(b,n,-1)+self.embedding)
        score=self.q(h)@self.k(h).transpose(1,2)/math.sqrt(h.shape[-1])
        a=select_edges(score,self.mode,self.tau)
        message=a@self.value(h)
        gate=self.gate(h).sigmoid()
        correction=self.correction(torch.cat([h,message],-1))-self.correction(torch.cat([h,torch.zeros_like(message)],-1))
        strength=self.direct(h)+gate*correction
        return dict(rhat=usd_pin(strength[:,:,0]),adj=a,message=message,graph_gate=gate,
                    edge_logits=score,regularization={},components={'rel':correction})


class OriginalSelector(StableRelationalScoreGraphFX):
    def __init__(self,dims,currencies,mode,tau=0.):
        super().__init__(n_ccy=len(currencies),currency_names=currencies,dims=dims,hidden=48,
                         top_k=3,graph_rank=8,dropout=.25,edge_dropout=.05,spectral_bound=1.)
        self.mode=mode;self.tau=tau;self._message=None

    def _build_graph(self,h,h_global,A_prev=None,A_override=None):
        # Same original static + node-state QK + macro-conditioned low-rank score.
        b,n,hdim=h.shape
        static_norm=self._normalize_static_adj()
        rank_state=self.rank_mlp(h_global)
        delta=torch.einsum('ir,br,jr->bij',self.node_left_embedding,rank_state,self.node_right_embedding)
        score=self.static_edge_logits.unsqueeze(0)+(self.Wq(h)@self.Wk(h).transpose(1,2))/math.sqrt(hdim)+delta
        a=self.edge_dropout(select_edges(score,self.mode,self.tau,self.top_k))
        a=torch.nan_to_num(a,nan=0.,posinf=0.,neginf=0.)
        effective=a if A_override is None else A_override
        message=effective@self.Wv(h);self._message=message
        h_rel=self.rel_norm(h+self.message_proj(message))
        raw=self.rel_head(h_rel).squeeze(-1)
        global_node=h_global.unsqueeze(1).expand(-1,n,-1)
        gate=self.graph_gate_mlp(torch.cat([h,global_node,h_rel],-1)).sigmoid().squeeze(-1)
        eye=torch.eye(n,dtype=torch.bool,device=h.device).unsqueeze(0)
        return dict(adj=effective,adj_current=a,adj_static=static_norm,adj_dynamic_delta=delta,
             edge_logits=score.masked_fill(eye,-torch.inf),
             edge_contrib=effective*self.rel_value_head(h).squeeze(-1)[:,None,:],
             graph_gate=gate,rel_repr=h_rel,rel_score=gate*raw,
             regularization=self.graph_regularization(effective,A_prev))

    def forward(self,batch):
        out=super().forward(batch)
        out['message']=self._message
        return out
