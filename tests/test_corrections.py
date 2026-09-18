import pytest
import numpy as np 
from src.corrections import bonferroni,benjamini_hochberg

@pytest.mark.parametrize("correction", [bonferroni, benjamini_hochberg])
def test_correction_returns_mask_of_input_length(correction):
    p_values = [0.01,0.02,0.03,0.04,0.05,0.06,0.7]
    assert correction(p_values).shape == (7,)

@pytest.mark.parametrize("correction", [bonferroni, benjamini_hochberg])
def test_correction_returns_boolean_array(correction):
    p_values = [0.01,0.02,0.03,0.04,0.05,0.06,0.7]
    assert correction(p_values).dtype == bool

@pytest.mark.parametrize("correction", [bonferroni, benjamini_hochberg])
def test_correction_accepts_lists_and_arrays(correction):
      p_values = [0.01,0.02,0.03,0.04,0.05,0.06,0.7]
      p_arr = np.asarray(p_values)
      
      assert np.array_equal(correction(p_arr), correction(p_values))
       
@pytest.mark.parametrize("correction", [bonferroni, benjamini_hochberg])
def test_correction_rejects_nothing_when_all_p_values_are_large(correction):
    p_values = [0.6,0.7,0.9,0.8,1]
    assert not(correction(p_values).any() == True)

def test_bonferroni_rejects_at_or_below_threshold():
    p_values = [0.004,0.002,0.003,0.005,0.006,0.2,0.3,0.5,0.6,1]
    mask = [True, True, True, True, False, False, False, False, False, False]
    assert np.array_equal(mask, bonferroni(p_values))

def test_benjamini_hochberg_rejects_everything_below_the_cutoff_rank():
    p_values = np.asarray([0.002,0.025,0.027,0.05,0.1])
    mask = np.asarray([True, True, True, False, False])
    
    order = [4, 2, 0, 1, 3]
    
    assert np.array_equal(benjamini_hochberg(p_values[order]), mask[order])
    
def test_bh_rejections_contain_bonferroni_rejections():
    rng1 = np.random.default_rng(10)
    a_tuple = (0.01, 0.05, 0.1, 0.2)
    for run_count in range(100):
        m = rng1.integers(1,250)
        alpha = a_tuple[run_count%4]
        p_values = rng1.uniform(0,1,size = m)
        
        bon_mask = bonferroni(p_values, alpha)
        ben_mask = benjamini_hochberg(p_values, alpha)
        
        assert ben_mask[bon_mask].all()
        