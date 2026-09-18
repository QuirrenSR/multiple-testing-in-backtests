import numpy as np

def bonferroni(p_values, alpha = 0.05) -> np.ndarray:
    """Checks the p_values of a given outcome and using the Bonferroni criterion returns a boolean array.

    Parameters
    ----------
    p_values : array_like
        P values of individual tests.
    alpha : float, optional
        Standard significance level for tests. 

    Returns
    -------
    np.ndarray
        Boolean array. True values reject the null hypothesis.
    """
    p_arr = np.asarray(p_values)
    m = len(p_arr)
    return p_arr <= alpha/m

def benjamini_hochberg(p_values, alpha = 0.05) -> np.ndarray:
    #Initialize needed variables
    p_arr = np.asarray(p_values)
    m = len(p_arr)
    thresholds = np.arange(1,m+1)/m *alpha
    
    #Create an ordering array so we remember which p_values are where in a sorted array.
    order = np.argsort(p_arr)
    
    #Create a boolean mask for sorted p_values and find the maximum that passes the constraint.
    passes = p_arr[order] <= thresholds
    largest = (np.nonzero(passes)[0]).max(initial = -1)
    
    #Write back a mask array for the p_values that passed the threshold test
    mask = np.zeros(m, dtype=bool)
    mask[order[:largest + 1]] = True
    
    return mask