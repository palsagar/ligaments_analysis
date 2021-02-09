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


# COMPUTATION OF ERROR BARS 

# Lower level function : computes mean heights of histogram for 1 ensemble of samples
def ensemble_heights(data, size_per_sample, n_samples , n_bins):

    import numpy as np
    
    ensemble_list = [] # create the empty list for creating the ensemble
    mean_heights_ensemble = []

    for _ in range(n_samples): 
        sample_n = np.random.choice(data, size = size_per_sample)
        heights_sample_n , bins_sample_n = np.histogram(sample_n, bins=n_bins)
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

def plot_histogram_error(data, size_per_sample, n_bins, heights, errors):

    import numpy as np
    import matplotlib.pyplot as plt

    sample_new = np.random.choice(data, size = size_per_sample)
    bins = np.linspace(0.0, np.amax(sample_new), n_bins + 1)
    bins_centers = np.array([0.5 * (bins[i] + bins[i+1]) for i in range(len(bins)-1)])

    total_area = 0.0

    for i in range(n_bins):
        total_area += heights[i]  

    bin_width = bins[1]-bins[0]
    total_area *= bin_width

    xspace = np.linspace(0.0, np.amax(sample_new), 10000)
    
    plt.figure(figsize=(8,8));
    plt.errorbar(bins_centers, heights/total_area, yerr= (1.96 * errors)/total_area, 
                marker = 'o', markersize = 1.5 , linestyle = 'none', 
                elinewidth = 0.5 , capsize=4.0 , capthick=0.5) ; 
    plt.yscale('log', basey=10);
    plt.ylabel("PDF", fontsize=14); 
    plt.xlabel("$D/W$", fontsize=14);

    return bins_centers, heights/total_area , xspace


def gauss(x, mu, sigma):
    import numpy as np

    return ( (1.0/(sigma*np.sqrt(2.0*np.pi))) * np.exp(-0.5 * ((x - mu)/sigma)**2) )

def gamma_n(x, a, b):
    from scipy.special import gamma
    import numpy as np
    return ( (b**a/gamma(a)) * x**(a - 1.0) * np.exp(-1.0 * b * x) )

def lognorm(x, mu_log, sigma_log):
    import numpy as np
    return ( (1.0/(x * sigma_log * np.sqrt(2.0*np.pi))) * np.exp(-0.5 * ( (np.log(x) - mu_log)/sigma_log)**2) )


def poisson(x, lam): 
    import numpy as np
    return (3.0 * x**2.0 * lam * np.exp(-1.0 * lam * x**3.0) )

def fit(x,y, func):
    from scipy.optimize import curve_fit

    popt, pcov = curve_fit(func, xdata=x, ydata=y)
    return popt, pcov
  




    


