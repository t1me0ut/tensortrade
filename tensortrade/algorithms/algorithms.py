import numpy as np


def simple_trend(data: np.ndarray) -> int:
    if len(data) < 2:
        return 0
    else:
        return np.sign(data[-1] - data[-2])
