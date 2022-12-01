# Copyright 2021 solo-learn development team.

# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to use,
# copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the
# Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies
# or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
# PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE
# FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import torch
import torch.nn.functional as F

def dual_temperature_loss_func(
    query: torch.Tensor,
    key: torch.Tensor, 
    temperature=0.1,
    dt_m=10,
) -> torch.Tensor:
    """
    query: anchor sample
    key: positive sample
    temperature: intra-anchor hardness-awareness control temperature
    dt_m: the scalar number to get inter-anchor hardness awareness temperature
    """

    # intra-anchor hardness-awareness
    b = query.size(0)
    pos = torch.einsum("nc,nc->n", [query, key]).unsqueeze(-1)

    # Selecte the intra negative samples according the updata time, 
    neg = torch.einsum("nc,ck->nk", [query, key.T])
    mask_neg = torch.ones_like(neg, dtype=bool)
    mask_neg.fill_diagonal_(False)
    neg = neg[mask_neg].reshape(neg.size(0), neg.size(1)-1)
    logits = torch.cat([pos, neg], dim=1)
    
    logits_intra = logits / temperature
    prob_intra = F.softmax(logits_intra, dim=1)

    # inter-anchor hardness-awareness
    logits_inter = logits / (temperature*dt_m)
    prob_inter = F.softmax(logits_inter, dim=1)

    # hardness-awareness factor
    inter_intra = (1 - prob_inter[:, 0]) / (1 - prob_intra[:, 0])

    loss = -torch.nn.functional.log_softmax(logits_intra, dim=-1)[:, 0]

    # final loss
    loss = inter_intra.detach() * loss
    loss = loss.mean()

    return loss
