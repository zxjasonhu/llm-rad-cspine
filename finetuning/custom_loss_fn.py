import torch
import torch.nn as nn
from typing import Optional


def _reshape_impression_weights(
    impression_weight: torch.Tensor,
    batch_size: int,
    seq_len: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """
    Support either one weight per example or one weight per token.
    """
    impression_weight = impression_weight.to(device=device, dtype=dtype)

    if impression_weight.numel() == batch_size:
        return impression_weight.view(batch_size, 1).expand(batch_size, seq_len)

    if impression_weight.numel() == batch_size * seq_len:
        return impression_weight.view(batch_size, seq_len)

    raise ValueError(
        "impression_weight must have either batch_size elements or "
        "batch_size * seq_len elements. "
        f"Got numel={impression_weight.numel()}, batch_size={batch_size}, seq_len={seq_len}."
    )


def _get_weighted_impression_loss_terms(
    source: torch.Tensor,
    target: torch.Tensor,
    impression_weight: torch.Tensor,
    batch_size: int,
    ignore_index: int = -100,
):
    token_loss = nn.functional.cross_entropy(
        source,
        target,
        ignore_index=ignore_index,
        reduction="none",
    ).view(batch_size, -1)
    valid_mask = target.ne(ignore_index).view(batch_size, -1).to(token_loss.dtype)
    impression_weights = _reshape_impression_weights(
        impression_weight,
        batch_size=batch_size,
        seq_len=token_loss.shape[1],
        device=token_loss.device,
        dtype=token_loss.dtype,
    )
    weighted_loss = token_loss * impression_weights * valid_mask

    return token_loss, valid_mask, impression_weights, weighted_loss


def _safe_denominator(value: torch.Tensor, min_value: float = 1.0) -> torch.Tensor:
    return value.clamp_min(min_value)


def weighted_impression_cross_entropy(
    source: torch.Tensor,
    target: torch.Tensor,
    impression_weight: torch.Tensor,
    batch_size: int,
    num_items_in_batch: Optional[torch.Tensor] = None,
    ignore_index: int = -100,
    **kwargs,
) -> torch.Tensor:
    """Normalize summed token loss by the total weighted token count."""
    _, valid_mask, impression_weights, weighted_loss = _get_weighted_impression_loss_terms(
        source,
        target,
        impression_weight,
        batch_size,
        ignore_index,
    )
    denominator = _safe_denominator((impression_weights * valid_mask).sum())
    return weighted_loss.sum() / denominator


def impression_weighted_causal_lm_loss(
    logits,
    labels,
    vocab_size: int,
    impression_weight: Optional[torch.Tensor] = None,
    batch_size: int = 1,
    num_items_in_batch: Optional[torch.Tensor] = None,
    ignore_index: int = -100,
    shift_labels: Optional[torch.Tensor] = None,
    **kwargs,
) -> torch.Tensor:
    if impression_weight is None:
        raise ValueError("impression_weight tensor is required")

    logits = logits.float()

    if shift_labels is None:
        labels = nn.functional.pad(labels, (0, 1), value=ignore_index)
        shift_labels = labels[..., 1:].contiguous()

    logits = logits.view(-1, vocab_size)
    shift_labels = shift_labels.view(-1).to(logits.device)

    return weighted_impression_cross_entropy(
        logits,
        shift_labels,
        impression_weight,
        batch_size,
        num_items_in_batch,
        ignore_index,
        **kwargs,
    )
