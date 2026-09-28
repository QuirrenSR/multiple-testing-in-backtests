import pytest
import numpy as np 
from src.experiment import max_sharpe_sweep, null_survival_counts


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
    
    
def test_ordering_theorem():
    alpha = (0.05, 0.2, 0.5)
    for i in range(12):
        rng = np.random.default_rng(1)
        counts = null_survival_counts(200,252,rng,alpha[i%3])
        assert (counts["bonferroni"] <= counts["benjamini_hochberg"] and counts["benjamini_hochberg"] <= counts["uncorrected"])
    
def test_alpha_equals_one():
    n_strategies = 50
    rng = np.random.default_rng(1)
    
    counts = null_survival_counts(n_strategies, 252, rng, alpha = 1)
    
    assert (counts["benjamini_hochberg"] == n_strategies) and (counts["uncorrected"] == n_strategies)
    
def test_uncorrected_count_binomial_band():
    rng = np.random.default_rng(0)
    n_strategies = 1000
    alpha = 0.05
    counts = null_survival_counts(n_strategies,252*4,rng,alpha= alpha)
    
    
    assert counts["uncorrected"] == pytest.approx(n_strategies * alpha, abs= 4* np.sqrt(n_strategies*alpha*(1- alpha)))