import numpy as np

def sharpe_ratio(returns, periods_per_year = 252) -> float:    
    """Calculates the Sharpe Ratio for a given array with return values. 
    
    Risk-free is taken as zero.

    Parameters
    ----------
    returns : array_like
        Array of periodic return values for a portfolio.
    periods_per_year : int, optional
        Periods per year to be used for calculation, by default 252

    Returns
    -------
    float
        Annualized Sharpe Ratio of the portfolio.

    Raises
    ------
    ValueError
        Invalid array size. Any returns arg needs to be size bigger than or equal to 2.
    ValueError
        Invalid variance in the array. The program expects there to be difference in the return array. If variance of the array is 0 this error is raised.
    """
    arr = np.asarray(returns)
    
    #Validity checks
    if len(arr) < 2:
        raise ValueError("Invalid array size.")
    
    std = np.std(arr, ddof=1)
    if std == 0:
        raise ValueError("Invalid variance in the array.")
    
    #Computation
    return (np.sqrt(periods_per_year) * np.mean(arr) / std)

def simulate_noise(n_periods, n_strategies, rng, sigma = 0.01) -> np.ndarray:
    """Simulates market noise for many strategies with no edge over many periods.


    Parameters
    ----------
    n_periods : int
        Number of periods to be simulated.
    n_strategies : int
        Number of different strategies to be simulated.
    rng : numpy.random.Generator
        Generator, the caller supplies it and simulate_noise doesn't create its own rng inside the function. This makes it so the noises are reproducible if need be.
    sigma : float, optional
       Standard deviation of the noise, by default 0.01 which approximates daily equity volatility.

    Returns
    -------
    np.ndarray
        A 2D array which has variates as its entries. Rows are periods, columns are strategies.
    """

    return rng.normal(loc = 0 , scale = sigma, size = (n_periods, n_strategies))
