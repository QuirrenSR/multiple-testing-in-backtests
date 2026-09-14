import numpy as np

def bonferroni(p_values, alpha) -> np.ndarray:
    p_arr = np.asarray(p_values)
    m = len(p_values)
    return p_arr <= alpha/m