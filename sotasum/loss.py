import torch


def label_smoothed_nll_loss_copy(lprobs, target, epsilon, ignore_index=None, sum=True):
    """From copy_is_all_you_need repo"""
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)
    nll_loss = -lprobs.gather(dim=-1, index=target)
    smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
    if ignore_index is not None:
        pad_mask = target.eq(ignore_index)
        nll_loss.masked_fill_(pad_mask, 0.)
        smooth_loss.masked_fill_(pad_mask, 0.)
    else:
        nll_loss = nll_loss.squeeze(-1)
        smooth_loss = smooth_loss.squeeze(-1)
    if sum:
        nll_loss = nll_loss.sum()
        smooth_loss = smooth_loss.sum()
    eps_i = epsilon / lprobs.size(-1)
    loss = (1. - epsilon) * nll_loss + eps_i * smooth_loss
    return loss, nll_loss


def label_smoothed_nll_loss_fairseq(lprobs, target, epsilon, ignore_index=None, reduce=True):
    """From fairseq"""
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)
    nll_loss = -lprobs.gather(dim=-1, index=target)
    smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
    if ignore_index is not None:
        pad_mask = target.eq(ignore_index)
        nll_loss.masked_fill_(pad_mask, 0.0)
        smooth_loss.masked_fill_(pad_mask, 0.0)
    else:
        nll_loss = nll_loss.squeeze(-1)
        smooth_loss = smooth_loss.squeeze(-1)
    if reduce:
        nll_loss = nll_loss.sum()
        smooth_loss = smooth_loss.sum()
    eps_i = epsilon / (lprobs.size(-1) - 1)
    loss = (1.0 - epsilon - eps_i) * nll_loss + eps_i * smooth_loss
    return loss, nll_loss


def label_smoothed_nll_loss_custom(
    lprobs: torch.Tensor,
    target: torch.Tensor,
    epsilon: float,
    ignore_index: int = None,
    reduce: str = None,
):
    """From fairseq"""
    eps_i = epsilon / (lprobs.size(-1) - 1)
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)
    nll_loss = -lprobs.gather(dim=-1, index=target)
    smooth_loss = -(lprobs * eps_i).sum(dim=-1, keepdim=True)
    if ignore_index is not None:
        pad_mask = target.eq(ignore_index)
        nll_loss.masked_fill_(pad_mask, 0.0)
        smooth_loss.masked_fill_(pad_mask, 0.0)
        count = (~pad_mask).sum()
    else:
        nll_loss = nll_loss.squeeze(-1)
        smooth_loss = smooth_loss.squeeze(-1)
    if reduce == "sum":
        nll_loss = nll_loss.sum()
        smooth_loss = smooth_loss.sum()
    elif reduce == "mean":
        nll_loss = nll_loss.sum() / count
        smooth_loss = smooth_loss.sum() / count
    loss = (1.0 - epsilon - eps_i) * nll_loss + smooth_loss
    return loss, nll_loss


def label_smoothed_nll_loss_transformers(
    log_probs: torch.Tensor,
    labels: torch.Tensor,
    epsilon: float,
    ignore_index: int,
):
    if labels.dim() == log_probs.dim() - 1:
        labels = labels.unsqueeze(-1)

    padding_mask = labels.eq(ignore_index)
    # In case the ignore_index is -100, the gather will fail, so we replace labels by 0. The padding_mask
    # will ignore them in any case.
    labels = torch.clamp(labels, min=0)
    nll_loss = log_probs.gather(dim=-1, index=labels)
    # works for fp16 input tensor too, by internally upcasting it to fp32
    smoothed_loss = log_probs.sum(dim=-1, keepdim=True, dtype=torch.float32)

    nll_loss.masked_fill_(padding_mask, 0.0)
    smoothed_loss.masked_fill_(padding_mask, 0.0)

    # Take the mean over the label dimensions, then divide by the number of active elements (i.e. not-padded):
    num_active_elements = padding_mask.numel() - padding_mask.long().sum()
    nll_loss = nll_loss.sum() / num_active_elements
    smoothed_loss = smoothed_loss.sum() / (num_active_elements *
                                           log_probs.shape[-1])
    return (1 - epsilon) * nll_loss + epsilon * smoothed_loss
