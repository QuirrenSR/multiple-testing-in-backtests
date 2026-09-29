import matplotlib.pyplot as plt
import numpy as np
from src.experiment import max_sharpe_sweep, null_survival_counts

if __name__ == "__main__":
    
    generator = np.random.default_rng(0)
    m = np.array([10,30,100,300,1000,3000])
    
    n_periods = 4*252
    n_trials = 200
    
    sweep = max_sharpe_sweep(m, n_periods, n_trials, generator)
    count_dict = null_survival_counts(1000, n_periods, generator)
    
    y = sweep.mean(axis = 1)
    yerr = sweep.std(axis = 1, ddof = 1)/np.sqrt(n_trials)
    
    
    grid = np.logspace(1,4,1000)
    curve_m1 = np.sqrt(2 * np.log(m)/ (n_periods/252))
    curve_m2 = (np.sqrt(2 * np.log(m)) - (np.log(np.log(m)) + np.log(4*np.pi))/(2 * np.sqrt(2*np.log(m))))/ np.sqrt(n_periods/252)
    curve_a = np.sqrt(2 * np.log(grid)/ (n_periods/252))
    curve_b = (np.sqrt(2 * np.log(grid)) - (np.log(np.log(grid)) + np.log(4*np.pi))/(2 * np.sqrt(2*np.log(grid))))/ np.sqrt(n_periods/252)
    
    
    
    print("Best-of-m annualised Sharpe: 4 years of daily returns, 200 trials per m, seed 0")
    print("    m     mean     SE      sqrt(2 ln m)    second-order")
    for i in range(len(m)):
        print(f"{m[i]:>6}{y[i]:>8.3f}{yerr[i]:>8.3f}{curve_m1[i]:>14.3f}{curve_m2[i]:>14.3f}")
        
        
    print("----------------------------------------------------------------")
    
    
    print("Survival Counts")
    print("Ran 1000 strategies over 4 years, with alpha = 0.05. By construction every survivor is a false positive.")
    label_dict = {"benjamini_hochberg" : "Benjamini-Hochberg", 
                  "bonferroni" : "Bonferroni",
                  "uncorrected" : "No Correction"}
    for key, value in count_dict.items():
        print(f"{label_dict[key]:<18}{value:>5}")       
    
    
    
    fig, ax = plt.subplots(figsize = (7,4.5))
    ax.errorbar(m, y, yerr = yerr, fmt = "o-", capsize= 3, color = "#2a78d6", linewidth = 2, markersize = 4, label = "Measured")
    ax.plot(grid, curve_a, linestyle = "--", linewidth = 2, color = "#eb6834", label = "√(2 ln m)")
    ax.plot(grid, curve_b, linestyle = ":", linewidth = 2, color = "#1baf7a", label = "√(2 ln m), second-order")
    ax.set_xscale("log")
    ax.set_xlabel("Strategies tested (m)")
    ax.set_ylabel("Annualised Sharpe of the best strategy")
    ax.set_title("Best-of-m Sharpe under a pure-noise null")
    ax.grid(True, which = "both", alpha = 0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig("figures/max_sharpe_vs_m.png", dpi = 150)