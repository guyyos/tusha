import pymc as pm
import numpy as np
import arviz as az
import pandas as pd
from pandas import DataFrame
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from scipy.signal import savgol_filter
from plot_creation import create_scatter_with_reg_line


def calc_smooth_hdi_line(x,hdi_data):
    x_data = np.linspace(x.min(), x.max(), 200)
    x_data[0] = (x_data[0] + x_data[1]) / 2
    hdi_interp = griddata(x, hdi_data, x_data)
    y_data = savgol_filter(hdi_interp, axis=0, window_length=55, polyorder=2)

    return x_data,y_data


def create_model_example1():
    RANDOM_SEED = 8927
    rng = np.random.default_rng(RANDOM_SEED)

    with pm.Model() as model:
        mu = pm.Normal("mu", mu=0, sigma=1)
        obs = pm.Normal("obs", mu=mu, sigma=1, observed=rng.standard_normal(100))
        idata = pm.sample()

    # return az.plot_trace(idata)
    return idata


def create_model_example2():
    d = pd.read_csv("data/Howell1.csv", sep=";", header=0)
    d2 = d[d.age >= 18]  # filter to get only adults


    # Get the average weight as part of the model definition
    xbar = d2.weight.mean()

    with pm.Model() as heights_model:

        # Priors are variables a, b, sigma
        # using pm.Normal is a way to represent the stochastic relationship the left has to right side of equation
        a = pm.Normal("a", mu=178, sigma=20)
        b = pm.Lognormal("b", mu=0, sigma=1)
        sigma = pm.Uniform("sigma", 0, 50)

        # This is a linear model (not really a prior or likelihood?)
        # Data included here (d2.weight, which is observed)
        # Mu is deterministic, but a and b are stochastic
        mu = pm.Deterministic('mu_height', a + b * (d2.weight - xbar))

        # Likelihood is height variable, which is also observed (data included here, d2.height))
        # Height is dependent on deterministic and stochastic variables
        height = pm.Normal("height", mu=mu, sigma=sigma, observed=d2.height)

        # The next lines is doing the fitting and sampling all at once.
        idata = pm.sample(1000, tune=1000)

    post = az.extract(idata)

    d2.plot('weight', 'height', kind='scatter')
    plt.plot(d2.weight, post.mean('sample')['mu_height'], 'C1')

    graph = az.plot_hdi(d2.weight, post['mu_height'].transpose());
    return graph


def create_numerical_univariates_model(df):
    def add_to_univariate_model(model, target):
        
        with model:
        # observed_data = pm.MutableData(target, df[target], dims='id')
            observed_data = pm.MutableData(target, df[target])

            mu = pm.Normal(f'mean[{target}]', mu=0, sigma=15)

            y = pm.Normal(f'y[{target}]',mu=mu, sigma=15, observed=observed_data)


    model = pm.Model()
    for col in df.columns:
        add_to_univariate_model(model, col)

    with model:
        idata = pm.sample()

    return idata


def create_numerical_univariate_model(df,target):
    model = pm.Model()

    param_name = f'mean[{target}]'
    with model:
        # observed_data = pm.MutableData(target, df[target], dims='id')
        observed_data = pm.MutableData(target, df[target])

        mu = pm.Normal(param_name, mu=0, sigma=15)

        y = pm.Normal(f'y[{target}]',mu=mu, sigma=15, observed=observed_data)

        idata = pm.sample()

    post = az.extract(idata)

    res = {param_name:post[param_name].values}

    return res



def create_categorical_univariate_model(df,target):
    observed_counts = df[target].value_counts().values

    k = len(df[target].unique())
    coords = {target: df[target].unique()}
    model = pm.Model(coords=coords)

    with model:
        p = pm.Dirichlet("p", a=np.ones(k),dims=(target,))
        counts = pm.Multinomial("counts", n=observed_counts.sum(), p=p, observed=observed_counts,dims=(target,))

        idata = pm.sample()

    post = az.extract(idata)

    res = {str(v.values):post['p'].sel({target:v}).values for v in post[target]}

    return res


def create_cat_to_num_model(df,target,cat_feature):
    cat_feature_idx,cat_feature_codes = pd.factorize(df[cat_feature])

    coords = {cat_feature: cat_feature_codes}

    mu_hyper_prior = df[target].mean()
    sd_hyper_prior = df[target].std()
    
    with pm.Model(coords=coords) as labeled_model:
        # hyper prior
        mu_hyper = pm.Normal('mu_hyper', mu=mu_hyper_prior, sigma=sd_hyper_prior)
        # per-group prior
        mu = pm.Normal('mu', mu=mu_hyper, sigma=sd_hyper_prior, dims=cat_feature)
        # likelihood
        likelihood = pm.Normal('likelihood', 
                            mu=mu[cat_feature_idx],
                            sigma=sd_hyper_prior, 
                            observed=df[target])

        idata = pm.sample()

    post = az.extract(idata)

    res = {feature_code:post['mu'].sel({cat_feature:feature_code}).values for feature_code in cat_feature_codes}
    return res


def transform_back(vals,mean,std):
    return vals*std+mean

def create_num_to_num_model(df,target,predictor):

    mean_target,std_target = df[target].mean(),df[target].std()
    df[target] = (df[target]-mean_target)/std_target

    mu_hyper_prior = 0 #df[target].mean()
    sd_hyper_prior = 1 #df[target].std()
    xbar = df[predictor].mean()

    with pm.Model() as model:

        a = pm.Normal("a", mu=mu_hyper_prior, sigma=sd_hyper_prior)
        b = pm.Lognormal("b", mu=0, sigma=1)
        sigma = pm.Uniform("sigma", 0, 50)

        mu = pm.Deterministic(f'mu_{target}', a + b * (df[predictor] - xbar))

        target_var = pm.Normal(target, mu=mu, sigma=sigma, observed=df[target])

        # The next lines is doing the fitting and sampling all at once.
        idata = pm.sample(1000, tune=1000)
        pm.sample_posterior_predictive(idata, extend_inferencedata=True)

    post = az.extract(idata)
    # post_predictive = az.extract(idata.posterior_predictive)
    hdi_prob = 0.9
    idata_hdi = az.hdi(idata, hdi_prob=hdi_prob)
    post_pred_hdi = az.hdi(idata.posterior_predictive, hdi_prob=hdi_prob)

    df[f'mu_{target}_lower'] = idata_hdi[f'mu_{target}'][:,0]
    df[f'mu_{target}_upper'] = idata_hdi[f'mu_{target}'][:,1]

    df[f'mu_{target}_mid'] = post.mean('sample')[f'mu_{target}']

    df[f'{target}_lower'] = post_pred_hdi[target][:,0]
    df[f'{target}_upper'] = post_pred_hdi[target][:,1]

    smooth_df = DataFrame()
    smooth_df[predictor],smooth_df[f'mu_{target}_hdi_lower'] = calc_smooth_hdi_line(df[predictor],df[f'mu_{target}_lower'].values)
    _,smooth_df[f'mu_{target}_hdi_upper'] = calc_smooth_hdi_line(df[predictor],df[f'mu_{target}_upper'].values)
    _,smooth_df[f'mu_{target}_mid'] = calc_smooth_hdi_line(df[predictor],df[f'mu_{target}_mid'].values)

    _,smooth_df[f'{target}_hdi_lower'] = calc_smooth_hdi_line(df[predictor],df[f'{target}_lower'].values)
    _,smooth_df[f'{target}_hdi_upper'] = calc_smooth_hdi_line(df[predictor],df[f'{target}_upper'].values)

    df[target] = transform_back(df[target],mean_target,std_target)
    smooth_df[f'mu_{target}_hdi_lower'] = transform_back(smooth_df[f'mu_{target}_hdi_lower'],mean_target,std_target)
    smooth_df[f'mu_{target}_hdi_upper'] = transform_back(smooth_df[f'mu_{target}_hdi_upper'],mean_target,std_target)
    smooth_df[f'mu_{target}_mid'] = transform_back(smooth_df[f'mu_{target}_mid'],mean_target,std_target)
    smooth_df[f'{target}_hdi_lower'] = transform_back(smooth_df[f'{target}_hdi_lower'],mean_target,std_target)
    smooth_df[f'{target}_hdi_upper'] = transform_back(smooth_df[f'{target}_hdi_upper'],mean_target,std_target)

    return create_scatter_with_reg_line(df,smooth_df,target,predictor)


# def create_univariate_model(df,target):
    
#     df1 = df.dropna()

#     param_name = f'mean[{target}]'
#     with pm.Model() as model_cont1:

#         # observed_data = pm.MutableData(target, df[target], dims='id')
#         observed_data = pm.MutableData(target, df1[target])

#         mu = pm.Normal(param_name, mu=0, sigma=15)

#         y = pm.Normal('y',mu=mu, sigma=15, observed=observed_data)
#         idata = pm.sample()

#     return idata,param_name


def find_parent(graph):
    for node,parents in graph.items():
        if len(parents)<=0:
            return node

def extract_node(graph):
    parent = find_parent(graph)
    #remove parent
    graph = {node:[p for p in parents if p!=parent] for node,parents in graph.items() if node!=parent}

    return parent,graph


def create_model(rv_graph,rv_types):
    with pm.Model() as model:

        vars = []

        graph = rv_graph.copy()

        rvs = {}

        while graph:
            node,graph = extract_node(graph)

            parents = rv_graph[node]
            if len(parents)>0:
                mu = pm.Deterministic(node+'_mu', np.sum([rvs[p] for p in parents]))

            else:
                mu = 0

            if rv_types[node] == 'StudentT':
                rvs[node] = pm.StudentT(node, nu=15, mu=mu, sigma=10)
            else:
                rvs[node] = pm.Normal(node, mu=mu, sigma=1)
    return model


def example_run1():
    rv_graph = {'a':['b','c','d'],'b':['c','d'],'c':['d'],'d':[]}
    rv_types = {'a':'StudentT','b':'Normal','c':'Normal','d':'Normal'}

    model = create_model(rv_graph,rv_types)

    with model:
            idata = pm.sample_prior_predictive(samples=1000, random_seed=rng)