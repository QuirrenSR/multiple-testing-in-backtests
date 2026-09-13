import pytest
import numpy as np 
from src.sharpe import sharpe_ratio, simulate_noise 

def test_sharpe_ratio_known_value():
    known = 0.02 / (0.0002)**(1/2)
    arr = [0.01,0.03]
    result = sharpe_ratio(arr, 1)
    assert known == pytest.approx(result)  
    
def test_sharpe_ratio_rejects_constant_list():
    with pytest.raises(ValueError):
        arr = [0.001,0.001,0.001]
        sharpe_ratio(arr)

def test_sharpe_ratio_rejects_short_list():
    with pytest.raises(ValueError):
        arr = [0.001]
        sharpe_ratio(arr)

def test_simulate_noise_depends_on_seed():
    
    rng0 = np.random.default_rng(0)
    rng1 = np.random.default_rng(1)
    
    sim0 = simulate_noise(10,100,rng0)
    sim1 = simulate_noise(10,100,rng1)
    
    assert not np.array_equal(sim0,sim1) 

def test_simulate_noise_returns_expected_shape():
    rng = np.random.default_rng(2)
    sim = simulate_noise(10,1000, rng)
    assert np.shape(sim) == (10,1000)
    
def test_sharpe_ratio_annualisation_scales_as_sqrt_periods():
    arr = [0.01,0.01,0.02,0.03]
    a = sharpe_ratio(arr, 1)
    b = sharpe_ratio(arr, 4)
    assert b == pytest.approx(a*2) 
    
def test_simulate_noise_is_reproducible():
    rng0 = np.random.default_rng(3)
    rng1 = np.random.default_rng(3)
    
    sim0 = simulate_noise(10,100,rng0)
    sim1 = simulate_noise(10,100,rng1)
    
    assert np.array_equal(sim0,sim1) 