import numpy as np
from src.sharpe import sharpe_ratio, simulate_noise
from src.corrections import bonferroni, benjamini_hochberg, sharpe_to_pvalue

def max_sharpe_sweep(m_values, n_periods, n_trials, rng) -> np.ndarray:
    """Find the max sharpe for multiple different number of strategies

    Every strategy has zero edge by construction; for each m the
    best of m annualised Sharpes is recorded, n_trials times; rows are the
    m values in input order, columns are trials

    Parameters
    ----------
    m_values : array_like
        Array of number of strategies to be simulated.
    n_periods : int
        Number of periods to be simulated.
    n_trials : int
        How many simulation trials to be run for each m value.
    rng : numpy.random.Generator
        Generator, the caller supplies it and max_sharpe_sweep doesn't create its own rng inside the function. This makes it so the noises are reproducible if need be.

    Returns
    -------
    np.ndarray
        A 2D array with best sharpe results for each trial and m_value. Rows are m_value indices, columns are trial numbers.
    """
    result_arr = np.zeros((len(m_values), n_trials))
    
    for m_idx, m_x in enumerate(m_values):
        for trial_x in range(n_trials):
            sim = simulate_noise(n_periods, m_x, rng)
            sharpes = []
            
            for i in range(sim.shape[1]):
                sharpes.append(sharpe_ratio(sim[:,i]))
                    
            result_arr[m_idx, trial_x] = max(sharpes)

    return result_arr