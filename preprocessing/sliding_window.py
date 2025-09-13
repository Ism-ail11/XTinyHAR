import numpy as np
from typing import Iterator, Tuple

def windows_indices(n: int, win: int, stride: int) -> Iterator[Tuple[int, int]]:
    """Yield (start, end) indices for sliding windows over length n."""
    if win <= 0 or stride <= 0:
        raise ValueError("win and stride must be positive")
    i = 0
    while i + win <= n:
        yield i, i + win
        i += stride
