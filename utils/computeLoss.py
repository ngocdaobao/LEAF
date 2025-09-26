import torch
import torch.nn.functional as F

def compute_CLLoss(Adj_mask, reprs, matsize, args, device): # compute InfoNCELoss
    logits_cl = torch.div(torch.matmul(reprs, reprs.T), args.cl_temp)
    if args.sub_max:
        logits_max_cl, _ = torch.max(logits_cl, dim=-1, keepdim=True)
        logits_cl = logits_cl - logits_max_cl
    exp_logits_cl = torch.exp(logits_cl)
    denom_cl = torch.sum(exp_logits_cl * (1 - torch.eye(matsize).to(device)), dim = -1) 
    log_prob_cl = -torch.mean((logits_cl - torch.log(denom_cl)) * Adj_mask, dim=-1)
    return torch.mean(log_prob_cl)

def CrossEntropyLossWithWeight(ce_outputs, ce_y, alpha = 0.3):
    per_sample_loss = F.cross_entropy(ce_outputs, ce_y, reduction='none')  # [batch_size]

    mask0      = (ce_y == 0)    # label 0
    mask_other = (ce_y != 0)    # label != 0

    if mask0.any():
        loss_group0 = per_sample_loss[mask0].mean()
    else:
        loss_group0 = torch.tensor(0., device=ce_outputs.device)

    if mask_other.any():
        loss_group_other = per_sample_loss[mask_other].mean()
    else:
        loss_group_other = torch.tensor(0., device=ce_outputs.device)

    loss_ce = alpha * loss_group0 + (1.0 - alpha) * loss_group_other
    return loss_ce