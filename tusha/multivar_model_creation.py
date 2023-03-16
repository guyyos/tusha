import pymc as pm
import numpy as np
import arviz as az
import pandas as pd
from pandas import DataFrame
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from scipy.signal import savgol_filter
import pytensor.tensor as at
from identify_features import identify_cols,FeatureType
from base_def import COUNTER_FACTUAL_NUM_POINTS,COUNTER_FACTUAL_SMOOTH_NUM_POINTS,PredictionSummary


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


def create_num_to_features_model_old(df,target,predictors,col_types):
       
    def factorize(vals):
        cat_feature_idx,cat_feature_codes = pd.factorize(vals)
        cat_feature_codes_map = {c:i for c,i in zip(cat_feature_codes,range(len(cat_feature_codes)))}

        return {'cat_feature_idx':cat_feature_idx,'cat_feature_codes':cat_feature_codes,'cat_feature_codes_map':cat_feature_codes_map}
    
    categorical_predictors = [col for col,col_type in col_types.items() \
                              if col_type in set([FeatureType.BOOL,FeatureType.CATEGORICAL]) and col in predictors]
    numerical_predictors = [col for col,col_type in col_types.items() if col_type==FeatureType.NUMERICAL and col in predictors]

    cat_factors = {col:factorize(df[col]) for col in categorical_predictors}

    coords = {'numerical_predictors':numerical_predictors}
    coords.update({cat_col:cat_factors[cat_col]['cat_feature_codes'] for cat_col in categorical_predictors})

    cur_dims = ['numerical_predictors']

    with pm.Model(coords=coords) as model:
        pred = pm.MutableData("pred", df[numerical_predictors].values)
        
        a = pm.Normal("a", mu=0, sigma=1)
        b_global = pm.Normal("b_global", mu=0, sigma=1)
        sigma = pm.Uniform("sigma", 0, 20)

        #per-group trend
        b = {p:pm.Normal(f'b_{p}', mu=b_global, sigma=1, dims=cur_dims) for p in categorical_predictors}

        predb = {p:pm.Deterministic(f'predb_{p}', at.dot(pred,b[p])) for p in categorical_predictors}
        shape = predb[categorical_predictors[0]].shape

        target_var = pm.Normal(target, mu=a+np.sum([predb[p][cat_factors[p]['cat_feature_idx']] for p in categorical_predictors]),\
                                sigma=sigma, shape=shape, observed=df[target])
        
        idata = pm.sample(1000, tune=1000)
        pm.sample_posterior_predictive(idata, extend_inferencedata=True)
            
    # post = az.extract(idata)
    # pm.model_to_graphviz(model)

    return model,idata


def test_create_num_to_features_model():
    num_items = 100
    df = pd.DataFrame(np.random.randint(0,4,size=(num_items, 4)), columns=list('ABCD'))
    df['A'] = np.random.randint(0,3,size=(num_items, 3))
    df['C'] = np.random.randint(0,3,size=(num_items, 5))
    df['target']=np.random.randn(1,num_items)[0]
    df['X']=np.random.randn(1,num_items)[0]
    df['Y']=np.random.randn(1,num_items)[0]
    df['W']=np.random.randn(1,num_items)[0]
    df['Z']=np.random.randn(1,num_items)[0]

    target = 'target'
    categorical_predictors = ['A','B','C','D']#['A','B'] #list('ABCD')
    numerical_predictors = ['X','Y','W','Z']

    cat_num_map = {'A':['X'],'B':['Y']}

    cat_factors = {col:factorize(df[col]) for col in categorical_predictors}

    model,idata = create_num_to_features_model(df,target,categorical_predictors,numerical_predictors,cat_num_map,cat_factors)

    post = az.extract(idata)
    plot = pm.model_to_graphviz(model)

    return model,idata,post,plot

def create_num_to_features_model(df,target,categorical_predictors,numerical_predictors,cat_num_map,cat_factors):

    cat_pred_no_num = list(set(categorical_predictors).difference(set([p for p,nps in cat_num_map.items() if len(nps)>0])))
    num_pred_cat = list(set([v for vs in cat_num_map.values() for v in vs]))
    num_pred_no_cat = list(set(numerical_predictors).difference(num_pred_cat))

    print(f'cat_pred_no_num {cat_pred_no_num}')
    print(f'num_pred_no_cat {num_pred_no_cat}')


    coords = {'num_pred_cat':num_pred_cat,'num_pred_no_cat':num_pred_no_cat}
    coords.update({cat_col:cat_factors[cat_col]['cat_feature_codes'] for cat_col in categorical_predictors})

    with pm.Model(coords=coords) as model:
        data_cat = {p:pm.MutableData(f'data_cat[{p}]',cat_factors[p]['cat_feature_idx']) for p in categorical_predictors}
        data_numeric_cat = {p:pm.MutableData(f'data_nc[{p}]', df[p].values) for p in num_pred_cat}

        a = pm.Normal("a", mu=0, sigma=1)
        sigma = pm.Uniform("sigma", 0, 20)
        
        b_global = {f'bglob[{p}]':pm.Normal(f'bglob[{p}]', mu=0, sigma=1) for p in categorical_predictors}

        #per-group trend
        bcn = {p:{nmp:pm.Normal(f'bcn[{p}][{nmp}]', mu=b_global[f'bglob[{p}]'], sigma=1, dims=p) for nmp in cat_num_map[p]} for p in categorical_predictors if p in cat_num_map}
        bcat = {p:pm.Normal(f'bcat[{p}]', mu=b_global[f'bglob[{p}]'], sigma=1, dims=p) for p in cat_pred_no_num}

        mu_numeric = 0
        if len(num_pred_no_cat)>0:
            data_numeric = pm.MutableData('data_num', df[num_pred_no_cat].values)
            bnum = pm.Normal("bnum", mu=0, sigma=1,dims="num_pred_no_cat")
            # mu_numeric = pm.Deterministic(f'mu_numeric', at.dot(data_numeric,bnum))
            mu_numeric = at.dot(data_numeric,bnum)

        # all_trends = pm.Deterministic('all_trends',pm.math.sum([sum([bcn[p][nmp][data_cat[p]]*data_numeric_cat[nmp] for nmp in cat_num_map[p]]) 
        #                                                 for p in categorical_predictors if p in cat_num_map] ))
        all_trends = sum([sum([bcn[p][nmp][data_cat[p]]*data_numeric_cat[nmp] for nmp in cat_num_map[p]]) 
                                                        for p in categorical_predictors if p in cat_num_map])
        
        # all_trends2 = pm.Deterministic('all_trends2',sum([bcat[p][data_cat[p]] for p in cat_pred_no_num]))
        all_trends2 = sum([bcat[p][data_cat[p]] for p in cat_pred_no_num])

        mu = pm.Deterministic(f'mu_{target}', a+all_trends+all_trends2+mu_numeric)
        
        target_var = pm.Normal(target, mu=mu, sigma=sigma, observed=df[target],shape=mu.shape)
        

        idata = pm.sample(1000, tune=1000)
        pm.sample_posterior_predictive(idata, extend_inferencedata=True)

    # post = az.extract(idata)
    # pm.model_to_graphviz(model)

    return model,idata


def calc_counterfactual_predictor(df,model,idata,target,active_predictor,categorical_predictors,numerical_predictors,
                                      cat_num_map,categorical_predictors_vals,cat_factors):
    
    print(f'calc_counterfactual_predictor: active_predictor = {active_predictor}')

    num_pred_cat = list(set([v for vs in cat_num_map.values() for v in vs]))
    num_pred_no_cat = list(set(numerical_predictors).difference(num_pred_cat))

    if active_predictor in numerical_predictors:
        df_counterfactual = DataFrame({active_predictor:np.linspace(df[active_predictor].min(), df[active_predictor].max(),COUNTER_FACTUAL_NUM_POINTS)})
    else:
        cat_codes = cat_factors[active_predictor]['cat_feature_codes']

        df_counterfactual = DataFrame({active_predictor:cat_codes})

    for p in numerical_predictors:
        if p != active_predictor:
            df_counterfactual[p] = 0

    for p in categorical_predictors:
        if p != active_predictor:
            df_counterfactual[p] = cat_factors[p]['cat_feature_codes_map'][categorical_predictors_vals[p]]
    
    with model:
        for p in categorical_predictors:
            pm.set_data({f'data_cat[{p}]': df_counterfactual[p].values})

        for p in num_pred_cat:
            pm.set_data({f'data_nc[{p}]': df_counterfactual[p].values})

        if len(num_pred_no_cat)>0:
            pm.set_data({"data_num": df_counterfactual[num_pred_no_cat].values})

        # use the updated values and predict outcomes and probabilities:
        # thinned_idata = idata.sel(draw=slice(None, None, 5))

        idata_2 = pm.sample_posterior_predictive(
            idata,
            var_names=[target,f'mu_{target}'],
            return_inferencedata=True,
            predictions=True,
        )
        
    return idata_2,df_counterfactual[active_predictor]


def summarize_predictions(idata,predictor,target,predictor_vals,cat_codes):
    predictions = az.extract(idata,'predictions')

    target_pred = predictions[target]
    mu_pred = predictions[f'mu_{target}']

    target_hdi = az.hdi(idata.predictions)[target]
    mu_hdi = az.hdi(idata.predictions)[f'mu_{target}']
    mu_mean = az.extract(idata.predictions).mean('sample')[f'mu_{target}']

    ps = PredictionSummary(predictor,target_pred,mu_pred,target_hdi,mu_hdi,mu_mean,predictor_vals,cat_codes)

    return ps


def create_model(df,target,predictors):

    features = predictors + [target]
    col_types = identify_cols(df,features)

    categorical_features = [col for col,col_type in col_types.items() \
                            if col_type in set([FeatureType.BOOL,FeatureType.CATEGORICAL])]

    numerical_features = [f for f in features if col_types[f]==FeatureType.NUMERICAL]

    print(f'create_model features = {features}')
    print(f'create_model numerical_features = {numerical_features}')

    cat_num_map = {}

    return create_model_and_predictions(df,target,predictors,categorical_features,numerical_features,cat_num_map)


def factorize(vals):
    cat_feature_idx,cat_feature_codes = pd.factorize(vals, sort=True)
    cat_feature_codes_map = {c:i for c,i in zip(cat_feature_codes,range(len(cat_feature_codes)))}

    return {'cat_feature_idx':cat_feature_idx,'cat_feature_codes':cat_feature_codes,'cat_feature_codes_map':cat_feature_codes_map}

def create_model_and_predictions(df,target,predictors,categorical_features,numerical_features,cat_num_map):

    categorical_predictors = [col for col in categorical_features if col in predictors]
    numerical_predictors = [col for col in numerical_features if col in predictors]

    feature_means = {f:df[f].mean() for f in numerical_features}
    feature_stds = {f:df[f].std() for f in numerical_features}

    for f in numerical_features:
        df[f] = standardize_vec(df[f],feature_means[f],feature_stds[f])

    cat_factors = {col:factorize(df[col]) for col in categorical_predictors}

    model,idata = create_num_to_features_model(df,target,categorical_predictors,numerical_predictors,cat_num_map,cat_factors)

    categorical_predictors_vals = {p:cat_factor['cat_feature_codes'][0] for p,cat_factor in cat_factors.items()}

    # return model,idata,categorical_predictors,numerical_predictors,feature_means,feature_stds,cat_factors,categorical_predictors_vals

    def unstand(predictor,ps):
        if predictor in feature_means:
            ps.predictor_vals = unstandardize_vec(ps.predictor_vals,feature_means[predictor],feature_stds[predictor])
        if target in feature_means:
            ps.target_hdi = unstandardize_vec(ps.target_hdi,feature_means[target],feature_stds[target])
            ps.mu_hdi = unstandardize_vec(ps.mu_hdi,feature_means[target],feature_stds[target])
            ps.mu_mean = unstandardize_vec(ps.mu_mean,feature_means[target],feature_stds[target])
            ps.target_pred = unstandardize_vec(ps.target_pred,feature_means[target],feature_stds[target])
            ps.mu_pred = unstandardize_vec(ps.mu_pred,feature_means[target],feature_stds[target])


    def smooth(ps):
        ps.target_hdi = savgol_filter(ps.target_hdi, axis=0, window_length=COUNTER_FACTUAL_SMOOTH_NUM_POINTS, polyorder=2)
        ps.mu_hdi = savgol_filter(ps.mu_hdi, axis=0, window_length=COUNTER_FACTUAL_SMOOTH_NUM_POINTS, polyorder=2)
        ps.mu_mean = savgol_filter(ps.mu_mean, axis=0, window_length=COUNTER_FACTUAL_SMOOTH_NUM_POINTS, polyorder=2)


    def calc_counterfactual(active_predictor):
        idata_2,predictor_vals = calc_counterfactual_predictor(df,model,idata,target,active_predictor,
                                        categorical_predictors,numerical_predictors,
                                        cat_num_map,categorical_predictors_vals,cat_factors)
        cat_codes = cat_factors[active_predictor]['cat_feature_codes'] if active_predictor in cat_factors else []

        ps = summarize_predictions(idata_2,active_predictor,target,predictor_vals,cat_codes)
        
        return ps
        

    pred_res = {active_predictor:calc_counterfactual(active_predictor) \
                                                    for active_predictor in predictors}

    for predictor,ps in pred_res.items():
        unstand(predictor,ps)
        if predictor in numerical_predictors:
            smooth(ps)

    return pred_res