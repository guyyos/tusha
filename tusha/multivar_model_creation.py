import pymc as pm
import numpy as np
import arviz as az
import pandas as pd
from pandas import DataFrame
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from scipy.signal import savgol_filter
import pytensor.tensor as at



def standardize_vec(unstd_vec,mean,std):
    return (unstd_vec-mean)/std

def unstandardize_vec(std_vec,mean,std):
    return std_vec*std+mean

def create_num_to_nums_model(df,target,predictors):

    mu_hyper_prior = 0 #df[target].mean()
    sd_hyper_prior = 1 #df[target].std()
        
    with pm.Model(coords={"predictors": predictors}) as model:
        pred = pm.MutableData("pred", df[predictors].values)

        a = pm.Normal("a", mu=mu_hyper_prior, sigma=sd_hyper_prior)
        # b = pm.Lognormal("b", mu=0, sigma=1,dims="predictors")
        b = pm.Normal("b", mu=0, sigma=1,dims="predictors")
        sigma = pm.Uniform("sigma", 0, 50)

        mu = pm.Deterministic(f'mu_{target}', a + at.dot(pred,b))

        target_var = pm.Normal(target, mu=mu, sigma=sigma, shape=mu.shape, observed=df[target])

        # The next lines is doing the fitting and sampling all at once.
        idata = pm.sample(1000, tune=1000)
        pm.sample_posterior_predictive(idata, extend_inferencedata=True)
        
    post = az.extract(idata)

    return model,idata


def calc_counterfactual_predictor(df,model,idata,target,predictors,active_predictor):
    num_points = 100
    df_counterfactual = DataFrame({active_predictor:np.linspace(df[active_predictor].min(), df[active_predictor].max(),num_points)})
    for p in predictors:
        if p != active_predictor:
            df_counterfactual[p] = 0

    
    with model:
        # # update values of predictors:
        pm.set_data({"pred": df_counterfactual[predictors].values})
        # use the updated values and predict outcomes and probabilities:
        idata_2 = pm.sample_posterior_predictive(
            idata,
            var_names=[target,f'mu_{target}'],
            return_inferencedata=True,
            predictions=True,
        )
        
    target_hdi = az.hdi(idata_2.predictions)[target]
    mu_hdi = az.hdi(idata_2.predictions)[f'mu_{target}']
    mu_mean = az.extract(idata_2.predictions).mean('sample')[f'mu_{target}']

    return target_hdi,mu_hdi,mu_mean,df_counterfactual[active_predictor]


def create_model(df,target,predictors):

    features = predictors + [target]

    print(f'create_model features = {features}')
    feature_means = {f:df[f].mean() for f in features}
    feature_stds = {f:df[f].std() for f in features}

    for f in features:
        df[f] = standardize_vec(df[f],feature_means[f],feature_stds[f])

    model,idata = create_num_to_nums_model(df,target,predictors)

    print(f"create_model a =  {az.hdi(idata)['a']}")
    print(f"create_model b =  {az.hdi(idata)['b']}")

    def unstand(predictor,target_hdi,mu_hdi,mu_mean,predictor_vals):
        target_hdi = unstandardize_vec(target_hdi,feature_means[target],feature_stds[target])
        mu_hdi = unstandardize_vec(mu_hdi,feature_means[target],feature_stds[target])
        mu_mean = unstandardize_vec(mu_mean,feature_means[target],feature_stds[target])
        predictor_vals = unstandardize_vec(predictor_vals,feature_means[predictor],feature_stds[predictor])
        return (target_hdi,mu_hdi,mu_mean,predictor_vals)

    def smooth(predictor,target_hdi,mu_hdi,mu_mean,predictor_vals):
        target_hdi = savgol_filter(target_hdi, axis=0, window_length=55, polyorder=2)
        mu_hdi = savgol_filter(mu_hdi, axis=0, window_length=55, polyorder=2)
        mu_mean = savgol_filter(mu_mean, axis=0, window_length=55, polyorder=2)
        target_hdi = savgol_filter(target_hdi, axis=0, window_length=55, polyorder=2)
        return (target_hdi,mu_hdi,mu_mean,predictor_vals)


    all_res = {active_predictor:calc_counterfactual_predictor(df,model,idata,target,predictors,active_predictor) \
                                                    for active_predictor in predictors}

    all_res = {predictor:unstand(predictor,target_hdi,mu_hdi,mu_mean,predictor_vals) for \
               predictor,(target_hdi,mu_hdi,mu_mean,predictor_vals) in all_res.items()}
    all_res = {predictor:smooth(predictor,target_hdi,mu_hdi,mu_mean,predictor_vals) for \
               predictor,(target_hdi,mu_hdi,mu_mean,predictor_vals) in all_res.items()}

    return all_res