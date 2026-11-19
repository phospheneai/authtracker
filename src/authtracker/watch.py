"""at.watch: log how large each layer's gradient is while training.

Exploding gradients show up as huge values, vanishing ones as values near zero. The numbers are
ordinary metrics (`grad_norm/<layer>` and `grad_norm/total`), so they work on every backend.
"""
from __future__ import annotations

from typing import Any, Callable

from .bundle import _unwrap
from .errors import AuthTrackerError, ValidationError


def watch(model: Any, every: int, current_run: Callable[[], Any]) -> Callable[[], None]:
    """Start watching `model`. Returns a function that stops it."""
    import torch
    if every < 1:
        raise ValidationError("every must be 1 or more")
    if not hasattr(torch.nn.Parameter, "register_post_accumulate_grad_hook"):
        raise AuthTrackerError("at.watch needs PyTorch 2.1 or newer")
    model = _unwrap(model)
    names = [n for n, p in model.named_parameters() if p.requires_grad]
    current_run().check_metric_names([f"grad_norm/{n}" for n in names] + ["grad_norm/total"])

    state = {"step": -1, "at": 0, "norms": {}}      # step: training forward passes so far, minus one

    def write() -> None:
        norms, state["norms"] = state["norms"], {}
        if not norms:
            return
        values = torch.stack(list(norms.values())).tolist()
        metrics = {f"grad_norm/{n}": v for n, v in zip(norms, values)}
        metrics["grad_norm/total"] = sum(v * v for v in values) ** 0.5
        try:
            current_run().log_metrics(metrics, state["at"])
        except AuthTrackerError:                    # no run is open any more: training goes on
            pass

    def before_forward(module, args) -> None:
        if module.training:
            write()                                 # the gradients of the previous step
            state["step"] += 1

    def collect(name: str):
        def hook(param) -> None:
            if state["step"] % every == 0 and param.grad is not None:
                state["at"] = state["step"]
                state["norms"][name] = param.grad.detach().norm()
        return hook

    handles = [model.register_forward_pre_hook(before_forward)]
    handles += [p.register_post_accumulate_grad_hook(collect(n))
                for n, p in model.named_parameters() if p.requires_grad]

    def stop() -> None:
        for handle in handles:
            handle.remove()
        write()
    return stop
