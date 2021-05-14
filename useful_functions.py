def preamble():
    
    from IPython.core.interactiveshell import InteractiveShell
    # pretty print all cell's output and not just the last one
    InteractiveShell.ast_node_interactivity = "all"

    # Required libraries for data arrays, data manipulation, plotting etc
    import numpy as np
    import pandas as pd

    import seaborn as sns
    sns.set_context("poster")
    sns.set(rc={'figure.figsize': (16, 9.)})
    sns.set_style("whitegrid")

    import matplotlib.pyplot as plt
    #%matplotlib inline
    #%config InlineBackend.figure_format = 'retina'
    plt.rcParams['text.latex.preamble']=[r"\usepackage{lmodern}",r'\boldmath']

# selecting time slice ,outputs numpy array 
def time_slice(df, time):
    ''' For a given dataframe, and choice of time 
        it returns a 1D numpy array with the dataset '''

    array = df[df.time == time].to_numpy()
    return array 


def bin_selection(data, n_bins):

    import numpy as np

    bin_edges = np.linspace(0.0, np.amax(data), n_bins + 1)
    bin_centers = np.array([0.5 * (bin_edges[i] + bin_edges[i+1]) for i in range(len(bin_edges)-1)])

    return bin_edges , bin_centers


# COMPUTATION OF ERROR BARS 

# Lower level function : computes mean heights of histogram for 1 ensemble of samples
def ensemble_heights(data, size_per_sample, n_samples , n_bins):

    import numpy as np
    
    ensemble_list = [] # create the empty list for creating the ensemble
    mean_heights_ensemble = []

    fixed_bin_edges , fixed_bin_centers = bin_selection(data, n_bins)

    for _ in range(n_samples): 
        sample_n = np.random.choice(data, size = size_per_sample)
        heights_sample_n , bins_sample_n = np.histogram(sample_n, bins= fixed_bin_edges)
        ensemble_list.append(heights_sample_n)

    heights_ensemble = np.array(ensemble_list)
    
    for i in range(n_bins):
        mean_height_bin_n = np.mean(heights_ensemble[:,i])
        mean_heights_ensemble.append(mean_height_bin_n)

    return np.array(mean_heights_ensemble)

# Higher level function : bootstraps the mean heights over several ensembles,
# Computes distributions of "means" of histogram heights

def ensemble_error_heights(n_bootstrap, data , size_per_sample , n_samples, n_bins):

    import numpy as np
    
    bootstrap_list = []

    for i in range(n_bootstrap):
        mean_heights_ensemble_n = ensemble_heights(data , size_per_sample , n_samples, n_bins)
        bootstrap_list.append(mean_heights_ensemble_n)

    mean_heights_bootstrap = np.array(bootstrap_list)

    error_heights_bootstrap = []
    
    for i in range(n_bins):
        error_mean_height_bin_n = np.std(mean_heights_bootstrap[:,i])
        error_heights_bootstrap.append(error_mean_height_bin_n)

    standard_error_mean_heights = np.array(error_heights_bootstrap)

    return standard_error_mean_heights

def plot_histogram_error(data, n_bins, heights, errors):

    import numpy as np
    import matplotlib.pyplot as plt

    #sample_new = np.random.choice(data, size = size_per_sample)
    #bins = np.linspace(0.0, np.amax(sample_new), n_bins + 1)
    #bins_centers = np.array([0.5 * (bins[i] + bins[i+1]) for i in range(len(bins)-1)])

    fixed_bin_edges , fixed_bin_centers = bin_selection(data, n_bins)

    total_area = 0.0

    for i in range(n_bins):
        total_area += heights[i]  

    #bin_width = bins[1]-bins[0]
    bin_width = fixed_bin_edges[1] - fixed_bin_edges[0] 
    total_area *= bin_width

    xspace = np.linspace(0.0, np.amax(data), 10000)
    
    plt.errorbar(fixed_bin_centers, heights/total_area, yerr= (1.96 * errors)/total_area, 
                marker = 'o', markersize = 1.5 , linestyle = 'none', 
                elinewidth = 0.5 , capsize=4.0 , capthick=0.5, label = "Ensemble Averaged Histogram") ; 
    plt.yscale('log', basey=10);
    plt.ylabel("PDF", fontsize=14); 
    plt.xlabel("$D/W$", fontsize=14);

    return fixed_bin_centers, heights/total_area , xspace


def gauss(x, mu, sigma):
    import numpy as np
    return ( (1.0/(sigma*np.sqrt(2.0*np.pi))) * np.exp(-0.5 * ((x - mu)/sigma)**2) )

def test_exp(x, A, B, C):
    import numpy as np
    return ( A * np.exp(-x**B/C) )

def gauss_2(x, A, mu, sigma):
    import numpy as np
    return  A * np.exp(-(x-mu)**2.0 / (2.0*sigma**2.0) )

def gauss_d6(x, mu, sigma):
    import numpy as np
    return ( (1.0/(sigma*np.sqrt(2.0*np.pi))) * np.exp(-0.5 * ((x**3.0 - mu**3.0)/sigma)**2) )

def gamma_n(x, a, b):
    from scipy.special import gamma
    import numpy as np
    return ( (b**a/gamma(a)) * x**(a - 1.0) * np.exp(-1.0 * b * x) )

def lognorm(x, mu_log, sigma_log):
    import numpy as np
    return ( (1.0/(x * sigma_log * np.sqrt(2.0*np.pi))) * np.exp(-0.5 * ( (np.log(x) - mu_log)/sigma_log)**2) )


def poisson(x, A,B): 
    import numpy as np
    return (A * np.exp(-1.0 * B * x) )

def poisson_2(x, A, lam): 
    import numpy as np
    return (A * np.exp(-1.0 * lam * x**3.0) )

def poisson_vol(x, A, B): 
    import numpy as np
    return (A * x**2.0 * np.exp(-1.0 * B * x**3.0) )

def poisson_std(x, A, B): 
    import numpy as np
    return (A * np.exp(-1.0 * B * x))


def pareto(x,A,B):
    return A * x ** (-1.0 * B)


def linear(x, m, c):
    return m * x + c  


def fit(x,y, func):
    from scipy.optimize import curve_fit

    popt, pcov = curve_fit(func, xdata=x, ydata=y, maxfev=100000)
    return popt, pcov
  


def stratify_sampling(x, n_samples, stratify):
    """Perform stratify sampling of a tensor.
    
    parameters
    ----------
    x: np.ndarray or torch.Tensor
        Array to sample from. Sampels from first dimension.
        
    n_samples: int
        Number of samples to sample
        
    stratify: tuple of int
        Size of each subgroup. Note that the sum of all the sizes 
        need to be equal to `x.shape[']`.
    """
    import numpy as np 

    n_total = x.shape[0]
    assert sum(stratify) == n_total
    
    n_strat_samples = [int(i*n_samples/n_total) for i in stratify]
    cum_n_samples = np.cumsum([0]+list(stratify))
    sampled_idcs = []
    for i, n_strat_sample in enumerate(n_strat_samples):
        sampled_idcs.append(np.random.choice(range(cum_n_samples[i], cum_n_samples[i+1]), 
                                            replace=False, 
                                            size=n_strat_sample))
        
    # might not be correct number of samples due to rounding
    n_current_samples = sum(n_strat_samples)
    if  n_current_samples < n_samples:
        delta_n_samples = n_samples - n_current_samples
        # might actually resample same as before, but it's only for a few
        sampled_idcs.append(np.random.choice(range(n_total), replace=False, size=delta_n_samples))
        
    samples = x[np.concatenate(sampled_idcs), ...]
    
    return samples


def dummy():
    return "Check passed hahahah! "


