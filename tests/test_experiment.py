import pytest
import numpy as np 
from src.experiment import max_sharpe_sweep


def test_sweep_returns_expected_shape():
    rng = np.random.default_rng(0)
    m_vals = [3,4,5]
    trials = 4
    assert (len(m_vals), trials) == np.shape(max_sharpe_sweep(m_vals, 10,trials, rng))
    
def test_sweep_is_reproducible():
    rng1 = np.random.default_rng(0)
    rng2 = np.random.default_rng(0)
    
    m_vals = [3,4,5]
    trials = 4
    
    sweep1 = max_sharpe_sweep(m_vals, 10,trials, rng1)
    sweep2 = max_sharpe_sweep(m_vals, 10,trials, rng2)
    
    assert np.array_equal(sweep1, sweep2)
    
def test_sweep_trials_differ_within_a_row():
    m_vals = [4]
    trials = 6
    rng = np.random.default_rng(0)
    
    sweep = max_sharpe_sweep(m_vals, 10, trials, rng)
    
    assert np.unique(sweep).size == trials
    
def test_best_sharpe_grows_with_m():
    m_vals = [5, 500]
    trials = 30
    rng = np.random.default_rng(0)
    
    sweep = max_sharpe_sweep(m_vals, 10, trials, rng)

    assert np.mean(sweep[0]) < np.mean(sweep[1])
    
def test_mean_max_matches_order_statistic_theory():
    m = [100]
    trials = 200
    years = 4
    periods = years* 252
    rng = np.random.default_rng(0)
    
    sweep = max_sharpe_sweep(m, periods, trials, rng)
    
    # E[max of m iid N(0,1)] = ∫ x · m·φ(x)·Φ(x)^(m-1) dx  (order-statistic density);
    # 2.5076 at m = 100, via scipy.integrate.quad. Divided by sqrt(years) to annualise.
    E_max = 2.5076    
    assert np.mean(sweep) == pytest.approx(E_max * 1/np.sqrt(years), abs = 0.06)