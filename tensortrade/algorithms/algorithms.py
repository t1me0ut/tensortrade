import numpy as np


def simple_trend(data: np.ndarray) -> int:
    if len(data) < 2:
        return 0
    else:
        return np.sign(data[-1] - data[-2])


def mooving_avg(data: np.ndarray, memory_len: int = 100, filter_len: int = 15) -> int:
    if len(data) < filter_len:
        return 0
    else:
        if len(data) > memory_len:
            data = data[-memory_len:]

        data_avg = np.convolve(np.concatenate([data, data[-(filter_len-1)//2:]]), np.ones((filter_len,))/filter_len, mode='same')
        data_avg = data_avg[:-(filter_len-1)//2]

        if np.abs(data_avg[-1] - data[-1]) < data[-1]*0.005:
            diff = data[-5:-2] - data_avg[-5:-2]
            if np.all(diff > 0):
                return 1
            else:
                return 0
        else:
            return 0
