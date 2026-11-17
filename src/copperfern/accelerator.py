"""Training helpers used by Copperfern."""

from typing import Any, Dict, List


def accumulate_gradients(losses: List[float], window: int) -> List[float]:
    if window <= 0:
        raise ValueError("window must be positive")
    result = []
    running = 0.0
    for index, loss in enumerate(losses, 1):
        running += loss
        if index % window == 0:
            result.append(running / window)
            running = 0.0
    return result


def save_checkpoint(path: str, state: Dict[str, Any]) -> str:
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(repr(state))
    return path
