import pymc as pm
import numpy as np
import arviz as az
import pandas as pd
from pandas import DataFrame
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from scipy.signal import savgol_filter
import pytensor.tensor as at
from identify_features import identify_series
from base_def import COUNTER_FACTUAL_NUM_POINTS, COUNTER_FACTUAL_SMOOTH_NUM_POINTS, PredictionSummary
from base_def import FeatureInfo, DAGraph, FeatureType
from pymc.sampling import jax
import pytensor.tensor as pt
import xarray as xr
import random



def standardize_vec(unstd_vec, mean, std):
    return (unstd_vec-mean)/std


def unstandardize_vec(std_vec, mean, std):
    return std_vec*std+mean


def create_num_to_nums_model(df, target, predictors):

    mu_hyper_prior = 0  # df[target].mean()
    sd_hyper_prior = 1  # df[target].std()

    with pm.Model(coords={"predictors": predictors}) as model:
        pred = pm.MutableData("pred", df[predictors].values)

        a = pm.Normal("a", mu=mu_hyper_prior, sigma=sd_hyper_prior)
        # b = pm.Lognormal("b", mu=0, sigma=1,dims="predictors")
        b = pm.Normal("b", mu=0, sigma=1, dims="predictors")
        sigma = pm.Uniform("sigma", 0, 50)

        mu = pm.Deterministic(f'mu_{target}', a + at.dot(pred, b))

        target_var = pm.Normal(target, mu=mu, sigma=sigma,
                               shape=mu.shape, observed=df[target])

        # The next lines is doing the fitting and sampling all at once.
        idata = pm.sample(1000, tune=1000)
        pm.sample_posterior_predictive(idata, extend_inferencedata=True)

    post = az.extract(idata)

    return model, idata


def factorize(vals):
    cat_feature_idx, cat_feature_codes = pd.factorize(vals, sort=True)
    cat_feature_codes = list(cat_feature_codes)
    cat_feature_codes_map = {c: i for c, i in zip(
        cat_feature_codes, range(len(cat_feature_codes)))}

    return cat_feature_idx, cat_feature_codes, cat_feature_codes_map


def create_feature_info(vals):

    col_type = identify_series(vals)
    finfo = FeatureInfo(col_type)

    if col_type == FeatureType.NUMERICAL:
        finfo.mean = vals.mean()
        finfo.std = vals.std()
    else:
        finfo.cat_feature_idx, finfo.cat_feature_codes, finfo.cat_feature_codes_map = factorize(
            vals)

    return finfo


def create_complete_model(df, df_relations):
    relations = df_relations.apply(lambda x: (x['Cause'], x['Effect']), axis=1)
    features = set([f for ce in relations for f in ce])

    nodes_info = {feat: create_feature_info(df[feat]) for feat in features}

    graph = DAGraph(nodes_info, relations)

    for f,node in graph.nodes.items():
        if node.info.featureType == FeatureType.NUMERICAL:
            df[f] = standardize_vec(df[f],node.info.mean,node.info.std)

    topo_order = graph.topological_order()

    complete_model = init_complete_model(df,graph)
    cat_num_map_per_target = {}

    for n in topo_order:
        # skip roots
        if len(graph.nodes[n].parent_vars) <= 0:
            continue

        predictor_nodes = {p: graph.nodes[p] for p in graph.nodes[n].parent_vars}
        complete_model,cat_num_map_per_target[n] = create_sub_model(df, graph.nodes[n], predictor_nodes, complete_model)
    
    # return complete_model
        
    model, idata = sample_model(complete_model)
    
    res = calc_counterfactual_predictor(df, model, idata, graph, topo_order, cat_num_map_per_target)
    summary_res = summarize_all_predictions(res,graph)
    smooth_all_summary(graph,summary_res)

    return model,res,summary_res,graph


def init_complete_model(df,graph):

    with pm.Model() as model:
        coords = {}

        for name,node in graph.nodes.items():
            if node.info.featureType.is_categorical():
                coords[name] = node.info.cat_feature_codes
                vals = node.info.cat_feature_idx
            else:
                vals = df[name].values

            v = pm.MutableData(f'data_{name}', vals)

        model.add_coords(coords)

    return model


def create_cat_num_map(df, node, parent_nodes):
    categorical_nodes = [n for n in parent_nodes.values() if n.info.featureType.is_categorical()] 
    numerical_nodes = [n for n in parent_nodes.values() if not n.info.featureType.is_categorical()]

    numerical_node_vars = set([n.name for n in numerical_nodes])

    cat_num_map = {}
    # find direct relations between parent_nodes (confounders)
    for cn in categorical_nodes:
        nbr_vars = set(cn.parent_vars+cn.child_vars)
        nbr_num_vars = nbr_vars.intersection(numerical_node_vars)
        cat_num_map[cn] = nbr_num_vars

    return cat_num_map


def create_sub_model(df, target_node, predictor_nodes, complete_model):
    cat_num_map = create_cat_num_map(df, target_node, predictor_nodes)
    add_sub_model(df, target_node, predictor_nodes, cat_num_map,complete_model)

    return complete_model,cat_num_map


def dot_vecs(vec1,vec2):
    # return at.dot(vec1,vec2)
    print(f'vec1 = {vec1}')
    print(f'vec2 = {vec2}')

    print(f'vec1.shape.eval() = {vec1.shape.eval()}')
    print(f'vec2.shape.eval() = {vec2.shape.eval()}')

    # print(f'vec1.eval() = {vec1.eval()}')
    # print(f'vec2.eval() = {vec2.eval()}')

    if len(vec1.shape.eval())<=0 or len(vec2.shape.eval())<=0:
        return at.dot(vec1,vec2)

    mtx1 = pt.broadcast_to(vec1, (1,vec1.shape[0])).T
    mtx2 = pt.broadcast_to(vec2, (1,vec2.shape[0]))

    return at.dot(mtx1,mtx2)


def add_sub_model(df, target_node, predictor_nodes, cat_num_map,model):
    target = target_node.name

    categorical_predictors = [name for name, n in predictor_nodes.items() if n.info.featureType.is_categorical()]
    numerical_predictors = [name for name, n in predictor_nodes.items() if not n.info.featureType.is_categorical()]

    cat_pred_no_num = list(set(categorical_predictors).difference(
        set([p for p, nps in cat_num_map.items() if len(nps) > 0])))
    num_pred_cat = list(set([v for vs in cat_num_map.values() for v in vs]))
    num_pred_no_cat = list(set(numerical_predictors).difference(num_pred_cat))

    print(f'cat_pred_no_num {cat_pred_no_num}')
    print(f'num_pred_no_cat {num_pred_no_cat}')

    dim_target = target if target_node.info.featureType == FeatureType.CATEGORICAL else None

    print(f'dim_target {dim_target}')

    with model:

        a = pm.Normal(f'a_{target}', mu=0, sigma=1)

        b_global = {f'bglob_{target}[{p}]': pm.Normal(
            f'bglob_{target}[{p}]', mu=0, sigma=1,dims=dim_target) for p in categorical_predictors}

        # per-group trend
        bcn = {p: {nmp: pm.Normal(f'bcn_{target}[{p}][{nmp}]', 
                                  mu=b_global[f'bglob_{target}[{p}]'], sigma=1, dims=(p,dim_target) if dim_target else p)
                   for nmp in cat_num_map[p]}
               for p in categorical_predictors if p in cat_num_map}

        bcat = {p: pm.Normal(
            f'bcat_{target}[{p}]', mu=b_global[f'bglob_{target}[{p}]'], 
            sigma=1, dims=(p,dim_target) if dim_target else p) for p in cat_pred_no_num}

        bnum = {p: pm.Normal(
            f'bnum_{target}[{p}]', mu=0, sigma=1, dims=dim_target) for p in num_pred_no_cat}

        all_trends_cn = sum([sum([dot_vecs(bcn[p][nmp][model.named_vars[f'data_{p}']],model.named_vars[f'data_{nmp}']) for nmp in cat_num_map[p]])
                          for p in categorical_predictors if p in cat_num_map])

        all_trends_c = sum([bcat[p][model.named_vars[f'data_{p}']] for p in cat_pred_no_num])

        all_trends_n = sum([dot_vecs(model.named_vars[f'data_{p}'],bnum[p]) for p in num_pred_no_cat])

        print(f'all_trends_cn = {all_trends_cn}')
        print(f'all_trends_c = {all_trends_c}')
        print(f'all_trends_n = {all_trends_n}')

        mu = pm.Deterministic(
            f'mu_{target}', a+all_trends_cn+all_trends_c+all_trends_n)
        
        print(f'mu.shape = {mu.shape.eval()}')
        print(f'mu.eval = {mu.eval()}')

        if target_node.info.featureType == FeatureType.BOOL:

            likelihood = pm.invlogit(mu)
            print(f'likelihood.shape = {likelihood.shape.eval()}')

            target_var = pm.Bernoulli(target,likelihood,shape=likelihood.shape,
                observed=model.named_vars[f'data_{target}'])

        elif target_node.info.featureType == FeatureType.CATEGORICAL:

            p = pm.Deterministic(f'p_{target}',pm.math.softmax(mu,axis=-1))
            print(f'p.shape = {p.shape.eval()}')
            print(f'p.shape = {p.shape.eval()}')
            print(f'p.eval = {p.eval()}')
            target_var = pm.Categorical(target,p=p, shape=p.shape[0], observed=model.named_vars[f'data_{target}'])
            
            #extremly slow!!
            # target_var = pm.Categorical(target,p=p, shape=p.shape, observed=model.named_vars[f'data_{target}'])

        else:
            sigma = pm.Uniform(f'sigma_{target}', 0, 20)
            print(f'mu.shape = {mu.shape.eval()}')

            target_var = pm.Normal(target, mu=mu, sigma=sigma,shape=mu.shape,
                        observed=model.named_vars[f'data_{target}'])

NUM_SAMPLES_PER_PREDICTION = 10

def create_vals_target_predictor(idata,target):
    return az.extract(idata.predictions, num_samples=NUM_SAMPLES_PER_PREDICTION)[target].values
    # return az.extract(idata.predictions, num_samples=NUM_SAMPLES_PER_PREDICTION)[target][:,0]

    # need to change below to handle categorical 
    # return az.extract(idata.predictions,num_samples=10).mean('sample')[target].values  

#need to block only upstream nodes from the predictor. donwstream predictors should be opne - therefore : None
# block: if numeric=0 else need to set every combination.
def calc_counterfactual_predictor(df, model, idata, graph, topo_order, cat_num_map_per_target):

    res = {}
    vals_target_predictors = {}

    for target in topo_order:
        res[target] = {}       
        vals_target_predictors[target] = {}

        for active_predictor in graph.nodes[target].parent_vars:

            cat_num_map = cat_num_map_per_target[target]

            #find the downstream vars
            downstream_predictor_vars = graph.get_downstream_vars(active_predictor)
            downstream_target_vars = graph.get_downstream_vars(target)

            vars_to_be_open = set(graph.nodes[target].parent_vars).intersection(downstream_predictor_vars)
            vars_to_be_blocked = set(graph.nodes[target].parent_vars).difference(vars_to_be_open.union({active_predictor}))
                    
            idata_2, predictor_vals = process_couterfactual_predictor(df, model, idata, graph, target, active_predictor,
                                cat_num_map,vars_to_be_open,vars_to_be_blocked, vals_target_predictors)

            res[target][active_predictor] = {}
            res[target][active_predictor]['idata'] = idata_2
            res[target][active_predictor]['predictor_vals'] = predictor_vals
    
    return res
            

def process_couterfactual_predictor(df, model, idata, graph, target, active_predictor,
                                cat_num_map,vars_to_be_open,vars_to_be_blocked,vals_target_predictors):

    print(
        f'process_couterfactual_predictor: target = {target}')
    print(
        f'process_couterfactual_predictor: active_predictor = {active_predictor}')
    print(
        f'process_couterfactual_predictor: cat_num_map = {cat_num_map}')
    print(
        f'process_couterfactual_predictor: vars_to_be_open = {vars_to_be_open}')
    print(
        f'process_couterfactual_predictor: vars_to_be_blocked = {vars_to_be_blocked}')

    if graph.nodes[active_predictor].info.featureType.is_categorical():
        cat_codes = [graph.nodes[active_predictor].info.cat_feature_codes_map[c] 
                    for c in graph.nodes[active_predictor].info.cat_feature_codes]

        df_counterfactual = DataFrame({active_predictor: cat_codes})
    else:
        df_counterfactual = DataFrame({active_predictor: np.linspace(
            df[active_predictor].min(), df[active_predictor].max(), COUNTER_FACTUAL_NUM_POINTS)})

    def inner_process_couterfactual_predictor(isample):
        for n in vars_to_be_blocked:
            node = graph.nodes[n]
            df_counterfactual[n] = random.choices(node.info.cat_feature_idx, k=len(df_counterfactual)) \
                                    if node.info.featureType.is_categorical() else 0

        for n in vars_to_be_open:
            df_counterfactual[n] = vals_target_predictors[n][active_predictor][:,isample]

        print('df_counterfactual')
        print(df_counterfactual)

        # var_names = list(set([target]).union(vars_to_be_open))
        # var_names = var_names+[f'mu_{v}' for v in var_names]
        var_names = [target,f'mu_{target}']
        print(f'process_couterfactual_predictor: var_names = {var_names}')

        with model:
            for n in df_counterfactual:
                pm.set_data({f'data_{n}': df_counterfactual[n].values})

            # use the updated values and predict outcomes and probabilities:
            thinned_idata = idata.sel(draw=slice(isample, None, 10))

            idata_2 = pm.sample_posterior_predictive(
                thinned_idata,
                var_names=var_names,
                return_inferencedata=True,
                predictions=True,
            )
        return idata_2

    if len(vars_to_be_open)>0:
        idata_new = None
        for isample in range(NUM_SAMPLES_PER_PREDICTION):
            idata_new1 = inner_process_couterfactual_predictor(isample)
            if idata_new:
                idata_new.predictions = xr.concat([idata_new.predictions,idata_new1.predictions],dim = 'draw')
            else:
                idata_new = idata_new1

    else:
        idata_new = inner_process_couterfactual_predictor(None)
    
    vals_target_predictors[target][active_predictor] = az.extract(idata_new.predictions, num_samples=NUM_SAMPLES_PER_PREDICTION)[target].values

    return idata_new, df_counterfactual[active_predictor]


def summarize_all_predictions(res,graph):
    summary_res = {}
    for target,res_target in res.items():
        summary_res[target] = {}
        for predictor,res_predictor in res_target.items():
            
            idata = res_predictor['idata']
            predictor_vals = res_predictor['predictor_vals']
            cat_codes = graph.nodes[predictor].info.cat_feature_codes\
                    if graph.nodes[predictor].info.featureType.is_categorical() \
                    else []

            prediction_summary = summarize_predictions(idata, predictor, target, predictor_vals, cat_codes)
            prediction_summary.predictors = graph.nodes[target].parent_vars
            summary_res[target][predictor] = prediction_summary
    
    return summary_res


def summarize_predictions(idata, predictor, target, predictor_vals, cat_codes):
    predictions = az.extract(idata, 'predictions')

    target_pred = predictions[target]
    mu_pred = predictions[f'mu_{target}']

    target_hdi = az.hdi(idata.predictions)[target]
    mu_hdi = az.hdi(idata.predictions)[f'mu_{target}']
    mu_mean = az.extract(idata.predictions).mean('sample')[f'mu_{target}']

    ps = PredictionSummary(predictor, target_pred, mu_pred,
                           target_hdi, mu_hdi, mu_mean, predictor_vals, cat_codes)

    return ps

def create_smooth_cat_target_data(prediction_summary,target,predictor,target_info,predictor_info):
    target_vals = prediction_summary.target_pred.to_numpy()
    num_samples = target_vals.shape[1]
    data = []
    for cat_idx,cat_val in enumerate(target_info.cat_feature_codes):

        target_val_counts = np.count_nonzero(target_vals == cat_idx,axis=1)
        if predictor_info.featureType.is_numerical():
            target_val_counts = savgol_filter(target_val_counts, axis=0, window_length=len(target_val_counts), polyorder=2)
            target_val_counts = [c if c>0 else 0 for c in target_val_counts]

        predictor_vals = prediction_summary.cat_codes if predictor_info.featureType.is_categorical() \
                                                    else prediction_summary.predictor_vals
            
        for pv,tv in zip(predictor_vals,target_val_counts):
            data.append({predictor:pv,'count':tv,target:cat_val})

    df_cat_target_data = DataFrame(data)
    df_cat_target_data['%'] = df_cat_target_data['count'] / df_cat_target_data.groupby(predictor)['count'].transform('sum')

    return df_cat_target_data

def smooth_all_summary(graph,summary_res):

    def unstand(predictor, ps,target_mean,target_std,predictor_mean,predictor_std):
        if predictor_mean is not None:
            ps.predictor_vals = unstandardize_vec(
                ps.predictor_vals, predictor_mean, predictor_std)
        if target_mean is not None:
            ps.target_hdi = unstandardize_vec(
                ps.target_hdi, target_mean,target_std)
            ps.mu_hdi = unstandardize_vec(
                ps.mu_hdi, target_mean,target_std)
            ps.mu_mean = unstandardize_vec(
                ps.mu_mean, target_mean,target_std)
            ps.target_pred = unstandardize_vec(
                ps.target_pred, target_mean,target_std)
            ps.mu_pred = unstandardize_vec(
                ps.mu_pred, target_mean,target_std)

    def smooth(ps):
        ps.target_hdi = savgol_filter(
            ps.target_hdi, axis=0, window_length=COUNTER_FACTUAL_SMOOTH_NUM_POINTS, polyorder=2)
        ps.mu_hdi = savgol_filter(
            ps.mu_hdi, axis=0, window_length=COUNTER_FACTUAL_SMOOTH_NUM_POINTS, polyorder=2)
        ps.mu_mean = savgol_filter(
            ps.mu_mean, axis=0, window_length=COUNTER_FACTUAL_SMOOTH_NUM_POINTS, polyorder=2)


    for target,res_target in summary_res.items():

        for predictor,prediction_summary in res_target.items():
            target_mean = target_std = predictor_mean = predictor_std = None
            target_info = graph.nodes[target].info
            predictor_info = graph.nodes[predictor].info

            print(f'smooth_all_summary {predictor} -> {target} ')

            if target_info.featureType == FeatureType.NUMERICAL:
                target_mean = target_info.mean
                target_std = target_info.std

            if predictor_info.featureType == FeatureType.NUMERICAL:
                predictor_mean = predictor_info.mean
                predictor_std = predictor_info.std

            unstand(prediction_summary.predictor, prediction_summary,
                    target_mean,target_std,predictor_mean,predictor_std)

            if predictor_info.featureType == FeatureType.NUMERICAL and target_info.featureType == FeatureType.NUMERICAL:
                smooth(prediction_summary)

            if target_info.featureType == FeatureType.CATEGORICAL:
                df_cat_target_data = create_smooth_cat_target_data(prediction_summary,target,predictor,target_info,predictor_info)
                prediction_summary.df_cat_target_data = df_cat_target_data





# categorical
    # age_d = pm.MutableData(f'age_d',df[['age']])
    # # adult_d = pm.MutableData('adult_d',df_adult.values)
    # adult_d = pm.MutableData('adult_d',df['adult'].values)

    # # thersh = pm.Normal(f'thersh', mu=50, sigma=10,dims='adult')

    # # alfa = pm.Normal('alfa', mu=0, sigma=1, shape=(1,len(cat_feature_codes)))#dims='adult'
    # beta = pm.Normal('beta', mu=0, sigma=1, shape=(1,len(cat_feature_codes)))
    # # mu = at.dot(age_d,beta)+alfa
    # # mu = beta*age_d+alfa
    # mu = pm.Deterministic('mu', at.dot(age_d,beta))
    # print(alfa.shape)
    # print(beta.shape)
    # print(age_d.shape)
    # print(df[['age']].shape)
    # print(mu.shape)

    # p = pm.Deterministic('p',pm.math.softmax(mu,axis=1))
    # # p = pm.math.softmax(mu,axis=1)#scipy.special.softmax(mu)

    # # adult = pm.Multinomial('adult',n=1,p=p, shape=mu.shape,observed=adult_d)
    # adult = pm.Categorical('adult',p=p, observed=adult_d)
    # # adult = pm.Categorical('adult',p=p, observed=None)
    # print(f'adult {adult.shape} {df_adult.values.shape}')

def sample_model(model):
    with model:
        idata = pm.sample(1000, tune=1000)
        # idata = jax.sample_numpyro_nuts(1000, tune=1000) # faster
        # idata = jax.sample_blackjax_nuts() # not working

        pm.sample_posterior_predictive(idata, extend_inferencedata=True)

    # post = az.extract(idata)
    # pm.model_to_graphviz(model)
    return model, idata

