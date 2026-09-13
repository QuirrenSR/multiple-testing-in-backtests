import numpy as np

def sharpe_ratio(returns, periods_per_year = 252) -> float:
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
    return rng.normal(loc = 0 , scale = sigma, size = (n_periods, n_strategies))