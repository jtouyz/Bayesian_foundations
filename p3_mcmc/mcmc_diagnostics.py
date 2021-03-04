# -- Import libraries --
import numpy as np
import pandas as pd
import plotnine as pn
from scipy.stats import norm, mode
import tensorflow_probability as tfp # for effective sample size

# Utility sampling functions
rnorm = lambda step_size: np.random.normal(scale = step_size, size = 1)
runif = lambda: np.random.uniform(size = 1)
rpois = lambda lambda_in : np.random.poisson(lambda_in, size = 1)
rbeta = lambda a,b: np.random.beta(a,b,size = 1)
rgamma = lambda a,b: np.random.gamma(shape = a, scale=1/b, size = 1)
rbinom = lambda n_in, theta_in: np.random.binomial(n_in, theta_in, size = 1)
cdfnorm = lambda l,s: norm.cdf(0,loc = l, scale = s)

# Needed to check that input is correct
class MyValidationError(Exception):
    "Class for validation input error"
    pass

# Defining the autocorrelation function
def acf(x, t=100):
    """
    Returns the autocorelation of x up to lag t

    Parameters
    ----------
    x : np.array
        Contains the samples from our mcmc sampler
    t : int
        Max lag to calculate the autocorrelation

    Returns
    -------
    out : pd.DataFrame
        Dataframe containing the autocorrelation and lag of input x
    """
    autocorrelations = np.array([1]+[np.corrcoef(x[:-i], x[i:])[0,1]  for i in range(1, t)])
    out = pd.DataFrame({'autocorrelation':autocorrelations}).reset_index().rename({'index':'lag'}, axis = 1)
    return(out)

# Autocorrelation plot
def plot_acf(data_in, figure_size = (15,5)):
    """
    Plots the autocorrelation function

    Parameteres
    -----------
    data_in : pd.DataFrame
        Dataframe containing the autcorrelation of our mcmc samples
    figure_size : tuple, default = (15,5)
        Optional input for figure size

    Returns
    -------
    pn.ggplot:
        Plotnine ggplot object containing autocorrelation plot
    """
    pn.options.figure_size = figure_size
    plot_out = pn.ggplot(pn.aes(x = 'lag', y = 'autocorrelation'), data = data_in)\
        + pn.geom_hline(pn.aes(yintercept= 0))\
        + pn.geom_hline(pn.aes(yintercept= 0.05), color = 'red', linetype = 'dashed')\
        + pn.geom_hline(pn.aes(yintercept= -0.05), color = 'red', linetype = 'dashed')\
        + pn.geom_col()
    return(plot_out)

# Trace plots
def plot_trace(data_in, figure_size = (15,5)):
    """
    Returns trace and density plot of mcmc samples from data_in.
    Note: the values 'chain', 'sample_i', 'parameter; and 'value' must be in the inputted pd.DataFrame

    Parameters
    ----------
    data_in : pd.DataFrame
        DataFrame containing samples from the sampler with columns: sample_i, chain, sample_i, and parameter
    figure_size : tuple, default = (15,5)
        Optional input for figure size
    
    Returns
    -------
    None:
        Prints out the trace and density plot for mcmc chains(s)
    """

    # Column validation
    name_check = set(data_in.columns)
    if name_check != set(['chain', 'sample_i', 'parameter', 'value' ]):
        raise MyValidationError("Incorrect column names in data_in please check")

    # Set figure size
    pn.options.figure_size = figure_size
    
    # Trace plot
    plot_out_trace = pn.ggplot(pn.aes(x = 'sample_i', y = 'value', color = 'chain'), data = data_in)\
        + pn.geom_line()\
        + pn.facet_grid('parameter ~ .')\
        + pn.labs(x = 'Sample', y = 'Parameter Value')
    
    # Distribution plot
    plot_out_distribution = pn.ggplot(pn.aes(x = 'value',
                                             color = 'chain'), data = data_in)\
        + pn.geom_density()\
        + pn.facet_grid('parameter ~ .')\
        + pn.labs(x = 'Parameter Value', y = 'Density')
    print(plot_out_trace)
    print(plot_out_distribution)
    return(None)

# MCMC diagnostics together
def mcmc_diagnostics(m_c, 
                     thin = 1, 
                     srf_details = False, 
                     plot_1_acf = False, 
                     plot_m_c_trace = False):
    """
    Wrapper function for returning mcmc diagnostic plots and summary measures for single or multiple chains

     Parameters
    ----------
    m_c : np.array
        Numpy array containing single or multiple chains from mcmc sampler
    thin : int
        Level of thinning
    srf_details : boolean  
        Display the details of the calculation for the scale reduction factor
    plot_1_acf : boolean
        Flag for whether to display the acf plot of the first chain?
    plot_m_c_trace : boolean
        Flag for whether to plot the trace and density plots for the chains?
    
    
    Returns
    -------
    pd.DataFrame:
        Contains summary measures for mcmc chains

    """
    # Thin sample?
    if thin>1:
        m_c = np.array([chain[::thin] for chain in m_c])
    
    # Constants
    n_chains, _ = m_c.shape

    # Reformatting chains
    col_names = ['chains_'+ str(i) for i in range(0,n_chains)]
    col_names = dict(zip(range(0,n_chains),col_names))
    m_c_df = pd.DataFrame(m_c.T).rename(col_names, axis = 1).reset_index().rename({'index':'sample_i'}, axis = 1)
    m_c_df =  pd.melt(m_c_df, id_vars = 'sample_i').assign(parameter = 'lambda').rename({'variable':'chain'}, axis = 1)

    # Plot 1 acf?
    if plot_1_acf:
        print(plot_acf(acf(m_c[0])))

    # Plot trace?
    if plot_m_c_trace:
        print(plot_trace(m_c_df))
    
    # Calculate effective sample size
    ess = tfp.mcmc.effective_sample_size(m_c.T, filter_beyond_positive_pairs=False).numpy().sum()
    print('The effective sampe size across our {a} chains are {b}'.format(a=n_chains, b=np.round(ess)))

    # -- Scale reduction factor --
    # Calculating the Within variance
    if srf_details:
        variance_within = m_c_df.groupby('chain').agg({'value':'var'})
        W = variance_within.value.mean()/n_chains
        print('The average variance within chains (W) is {}'.format(np.round(W,2)))

        # Calculating Between variance
        mean_individuals_chains =  m_c_df.groupby('chain').agg({'value':'mean'}).value
        B = np.var(mean_individuals_chains)
        print('The average variance between chains (B) is {}'.format(np.round(B,2)))

        # Calculating the the scale reduction factor
        V = W + B
        R = np.sqrt(V/W) # scale_redution_factor
        print('The average scale reduction (R) is {}'.format(np.round(R,2)))
    else:
        R = tfp.mcmc.potential_scale_reduction(m_c.T, independent_chain_ndims=1)
        print('The average scale reduction (R) is {}'.format(np.round(R,2)))

    # -- Output mean value and variance of parameter --
    flat_m_c = np.round(m_c.flatten(),3)
    summary_vals = {'mean_parameter' : np.mean(flat_m_c),
    'mode_parameter' :  mode(flat_m_c).mode,
    'std_dev_parameter' : np.std(flat_m_c),
    '5_percentile' : np.percentile(flat_m_c,5),
    '95_percentile': np.percentile(flat_m_c,95),
    'n_eff': np.ceil(np.mean(ess)),
    'R': np.round(R.numpy(),2)}
    return(pd.DataFrame(summary_vals))