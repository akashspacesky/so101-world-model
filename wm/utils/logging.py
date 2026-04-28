"""Simple console + WandB logger."""

from __future__ import annotations

import time
from typing import Any


class Logger:
    def __init__(self, project: str | None = None, config: dict | None = None):
        self._wandb = None
        if project:
            try:
                import wandb
                wandb.init(project=project, config=config or {})
                self._wandb = wandb
            except ImportError:
                pass
        self._last_print = 0.0

    def log(self, data: dict[str, Any], step: int | None = None, throttle_s: float = 0.0):
        now = time.time()
        if throttle_s and (now - self._last_print) < throttle_s:
            return
        self._last_print = now
        row = " | ".join(f"{k} {v:.4f}" if isinstance(v, float) else f"{k} {v}" for k, v in data.items())
        print(row)
        if self._wandb:
            self._wandb.log(data, step=step)

    def finish(self):
        if self._wandb:
            self._wandb.finish()
