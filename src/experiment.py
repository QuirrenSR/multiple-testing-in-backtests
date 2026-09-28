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



def null_survival_counts(n_strategies, n_periods, rng, alpha=0.05) -> dict:
    """Simulates strategies over a given number of periods, every strategy has zero edge by construction, so every survivor is a false positive.

    Parameters
    ----------
    n_strategies : int
        Number of different strategies to be simulated.
    n_periods : int
        Number of periods to be simulated. It is assumed that there are 252 periods in a single year.
    rng : numpy.random.Generator
        Generator, the caller supplies it and null_survival_counts doesn't create its own rng inside the function. This makes it so the noises are reproducible if need be.
    alpha : float, optional
        Standard significance level for tests, by default 0.05.

    Returns
    -------
    dict
        Three keys ("uncorrected", "bonferroni", "benjamini_hochberg"), each have values equal to the number of strategies the corrections thought were good strategies.
    """
    sim = simulate_noise(n_periods, n_strategies,rng)
    return_dict = {}
    sharpes = []
    
    for i in range(sim.shape[1]):
        sharpes.append(sharpe_ratio(sim[:,i]))
    
    p_vals = sharpe_to_pvalue(sharpes, n_periods/252)
       
    plain_mask = p_vals <= alpha
    bonferroni_mask = bonferroni(p_vals, alpha)
    benjamini_hochberg_mask = benjamini_hochberg(p_vals,alpha)
    
    return_dict["uncorrected"] = int(np.count_nonzero(plain_mask))
    return_dict["bonferroni"] = int(np.count_nonzero(bonferroni_mask))
    return_dict["benjamini_hochberg"] = int(np.count_nonzero(benjamini_hochberg_mask))
    
    return return_dict