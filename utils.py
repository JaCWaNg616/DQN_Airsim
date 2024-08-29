import numpy as np

def hash_state(state, bins):
    hashed = np.digitize(state, bins)
    return sum([hashed[i] * (len(bins) ** i) for i in range(len(bins))])
