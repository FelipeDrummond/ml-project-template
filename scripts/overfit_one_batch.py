"""Smoke test: overfit a single batch — loss must collapse toward zero.

If this script does NOT drive loss → ~0 within ~200 steps, the model wiring is
broken. This is the first thing to run after any change to model / loss / data
pipeline. Catches shape and device bugs in seconds, not hours.
"""

from __future__ import annotations

import torch
from torch import nn

from ml_template.models import MLP
from ml_template.utils import resolve_device, set_seed, setup_logging


def main() -> None:
    setup_logging("INFO")
    set_seed(0)
    device = resolve_device("auto")

    n_features, n_classes, batch_size = 16, 4, 32
    x = torch.randn(batch_size, n_features, device=device)
    y = torch.randint(0, n_classes, (batch_size,), device=device)

    model = MLP(
        in_features=n_features,
        out_features=n_classes,
        hidden_dim=64,
        n_layers=2,
    ).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=1e-2)
    loss_fn = nn.CrossEntropyLoss()

    model.train()
    losses: list[float] = []
    for step in range(200):
        logits = model(x)
        loss = loss_fn(logits, y)
        optim.zero_grad(set_to_none=True)
        loss.backward()
        optim.step()
        losses.append(loss.item())
        if step % 20 == 0:
            print(f"step {step:>3}  loss={loss.item():.4f}")

    final = losses[-1]
    print(f"\nfinal loss: {final:.6f}  (expected: < 0.01)")
    if final >= 0.01:
        raise SystemExit(f"smoke FAILED: final loss {final:.4f} >= 0.01")
    print("smoke OK")


if __name__ == "__main__":
    main()
