"""Training masks for the message stage: singleton, full and random-cardinality subsets of the 10 candidates."""
from __future__ import annotations

import torch

from .message_model import candidate_mask


def sample_masks(B, gen, mix=(.34, .33, .33)):
    """(B, 10, 11) float mask. For each (date, receiver) one subset type is drawn: singleton / full / random size 1..10."""
    cand = candidate_mask(B).float()
    u = torch.rand(B, 10, generator=gen)
    kind = (u > mix[0]).long() + (u > mix[0] + mix[1]).long()                      # 0 singleton, 1 full, 2 random
    score = torch.rand(B, 10, 11, generator=gen).masked_fill(cand == 0, 2.)       # excluded candidates rank last
    rank = score.argsort(-1).argsort(-1).float()                                  # 0..10
    k = torch.randint(1, 11, (B, 10), generator=gen).float()
    k = torch.where(kind == 0, torch.ones_like(k), torch.where(kind == 1, torch.full_like(k, 10.), k))
    return (rank < k.unsqueeze(-1)).float() * cand
