import torch
import numpy as np
from typing import Callable, Optional

@torch.no_grad()
def _linear_interpolate(baseline: torch.Tensor, input: torch.Tensor, alphas: torch.Tensor) -> torch.Tensor:
    # input shapes: baseline [B,W,C], input [B,W,C], alphas [m]
    alphas = alphas.view(-1, 1, 1, 1) # [m,1,1,1]
    baseline = baseline.unsqueeze(0)  # [1,B,W,C]
    input = input.unsqueeze(0)        # [1,B,W,C]
    return baseline + alphas * (input - baseline)

def integrated_gradients(
    model: torch.nn.Module,
    x: torch.Tensor,            # [B,W,C], requires_grad True
    target_fn: Callable[[torch.Tensor], torch.Tensor],  # logits-> scalar per sample
    baseline: Optional[torch.Tensor] = None,
    steps: int = 50
):
    """Compute Integrated Gradients for time-series input.
    Args:
        model: PyTorch model. Should return logits [B, K] from x.
        x: input batch [B, W, C].
        target_fn: maps logits [B,K] -> per-sample scalar (e.g., gather predicted logit).
        baseline: None -> zero baseline; or tensor [B,W,C]; or 'mean' -> mean(x) per-channel.
        steps: number of interpolation steps.
    Returns:
        attributions: [B, W, C] attribution map.
    """
    device = x.device
    model.eval()
    x = x.clone().detach().requires_grad_(True)

    if baseline is None or (isinstance(baseline, str) and baseline.lower() == "zero"):
        base = torch.zeros_like(x)
    elif isinstance(baseline, str) and baseline.lower() == "mean":
        base = x.mean(dim=1, keepdim=True).repeat(1, x.shape[1], 1).detach()
    else:
        base = baseline.to(device)

    alphas = torch.linspace(0.0, 1.0, steps=steps, device=device)
    scaled_inputs = _linear_interpolate(base, x, alphas)  # [m,B,W,C]

    grads_sum = torch.zeros_like(x)
    for i in range(steps):
        inp = scaled_inputs[i]
        inp.requires_grad_(True)
        logits = model(inp)[0] if isinstance(model(inp), tuple) else model(inp)  # support (logits, attn)
        out = target_fn(logits)  # [B]
        grads = torch.autograd.grad(outputs=out.sum(), inputs=inp, retain_graph=False, create_graph=False)[0]  # [B,W,C]
        grads_sum += grads

    avg_grads = grads_sum / steps
    attributions = (x - base) * avg_grads
    return attributions.detach()

def attribution_to_heatmap(attr: torch.Tensor) -> np.ndarray:
    """Convert [W,C] to heatmap [C,W] normalized for plotting."""
    a = attr.detach().cpu().numpy()  # [W,C]
    a = a.T  # [C,W]
    # min-max per-channel for visibility
    amin = a.min(axis=1, keepdims=True)
    amax = a.max(axis=1, keepdims=True)
    denom = (amax - amin) + 1e-8
    return (a - amin) / denom
