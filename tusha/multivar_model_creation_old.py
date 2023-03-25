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


def create_num_to_features_model_old(df, target, predictors, col_types):

    def factorize(vals):
        cat_feature_idx, cat_feature_codes = pd.factorize(vals)
        cat_feature_codes_map = {c: i for c, i in zip(
            cat_feature_codes, range(len(cat_feature_codes)))}

        return {'cat_feature_idx': cat_feature_idx, 'cat_feature_codes': cat_feature_codes, 'cat_feature_codes_map': cat_feature_codes_map}

    categorical_predictors = [col for col, col_type in col_types.items()
                              if col_type in set([FeatureType.BOOL, FeatureType.CATEGORICAL]) and col in predictors]
    numerical_predictors = [col for col, col_type in col_types.items(
    ) if col_type == FeatureType.NUMERICAL and col in predictors]

    cat_factors = {col: factorize(df[col]) for col in categorical_predictors}

    coords = {'numerical_predictors': numerical_predictors}
    coords.update({cat_col: cat_factors[cat_col]['cat_feature_codes']
                  for cat_col in categorical_predictors})

    cur_dims = ['numerical_predictors']

    with pm.Model(coords=coords) as model:
        pred = pm.MutableData("pred", df[numerical_predictors].values)

        a = pm.Normal("a", mu=0, sigma=1)
        b_global = pm.Normal("b_global", mu=0, sigma=1)
        sigma = pm.Uniform("sigma", 0, 20)

        # per-group trend
        b = {p: pm.Normal(f'b_{p}', mu=b_global, sigma=1, dims=cur_dims)
             for p in categorical_predictors}

        predb = {p: pm.Deterministic(f'predb_{p}', at.dot(
            pred, b[p])) for p in categorical_predictors}
        shape = predb[categorical_predictors[0]].shape

        target_var = pm.Normal(target, mu=a+np.sum([predb[p][cat_factors[p]['cat_feature_idx']] for p in categorical_predictors]),
                               sigma=sigma, shape=shape, observed=df[target])

        idata = pm.sample(1000, tune=1000)
        pm.sample_posterior_predictive(idata, extend_inferencedata=True)

    # post = az.extract(idata)
    # pm.model_to_graphviz(model)

    return model, idata


def create_num_to_features_model_old(df, target, categorical_predictors, numerical_predictors, cat_num_map, cat_factors):

    cat_pred_no_num = list(set(categorical_predictors).difference(
        set([p for p, nps in cat_num_map.items() if len(nps) > 0])))
    num_pred_cat = list(set([v for vs in cat_num_map.values() for v in vs]))
    num_pred_no_cat = list(set(numerical_predictors).difference(num_pred_cat))

    print(f'cat_pred_no_num {cat_pred_no_num}')
    print(f'num_pred_no_cat {num_pred_no_cat}')

    coords = {'num_pred_cat': num_pred_cat, 'num_pred_no_cat': num_pred_no_cat}
    coords.update({cat_col: cat_factors[cat_col]['cat_feature_codes']
                  for cat_col in categorical_predictors})

    with pm.Model(coords=coords) as model:
        data_cat = {p: pm.MutableData(
            f'data_cat[{p}]', cat_factors[p]['cat_feature_idx']) for p in categorical_predictors}
        data_numeric_cat = {p: pm.MutableData(
            f'data_nc[{p}]', df[p].values) for p in num_pred_cat}

        a = pm.Normal("a", mu=0, sigma=1)
        sigma = pm.Uniform("sigma", 0, 20)

        b_global = {f'bglob[{p}]': pm.Normal(
            f'bglob[{p}]', mu=0, sigma=1) for p in categorical_predictors}

        # per-group trend
        bcn = {p: {nmp: pm.Normal(f'bcn[{p}][{nmp}]', mu=b_global[f'bglob[{p}]'], sigma=1, dims=p)
                                  for nmp in cat_num_map[p]} for p in categorical_predictors if p in cat_num_map}
        bcat = {p: pm.Normal(
            f'bcat[{p}]', mu=b_global[f'bglob[{p}]'], sigma=1, dims=p) for p in cat_pred_no_num}

        mu_numeric = 0
        if len(num_pred_no_cat) > 0:
            data_numeric = pm.MutableData(
                'data_num', df[num_pred_no_cat].values)
            bnum = pm.Normal("bnum", mu=0, sigma=1, dims="num_pred_no_cat")
            # mu_numeric = pm.Deterministic(f'mu_numeric', at.dot(data_numeric,bnum))
            mu_numeric = at.dot(data_numeric, bnum)

        # all_trends = pm.Deterministic('all_trends',pm.math.sum([sum([bcn[p][nmp][data_cat[p]]*data_numeric_cat[nmp] for nmp in cat_num_map[p]])
        #                                                 for p in categorical_predictors if p in cat_num_map] ))
        all_trends = sum([sum([bcn[p][nmp][data_cat[p]]*data_numeric_cat[nmp] for nmp in cat_num_map[p]])
                          for p in categorical_predictors if p in cat_num_map])

        # all_trends2 = pm.Deterministic('all_trends2',sum([bcat[p][data_cat[p]] for p in cat_pred_no_num]))
        all_trends2 = sum([bcat[p][data_cat[p]] for p in cat_pred_no_num])

        mu = pm.Deterministic(
            f'mu_{target}', a+all_trends+all_trends2+mu_numeric)

        target_var = pm.Normal(target, mu=mu, sigma=sigma,
                               observed=df[target], shape=mu.shape)

        idata = pm.sample(1000, tune=1000)
        pm.sample_posterior_predictive(idata, extend_inferencedata=True)

    # post = az.extract(idata)
    # pm.model_to_graphviz(model)

    return model, idata



def calc_counterfactual_predictor_old(df, model, idata, target, active_predictor, categorical_predictors, numerical_predictors,
                                  cat_num_map, categorical_predictors_vals, cat_factors):

    print(
        f'calc_counterfactual_predictor: active_predictor = {active_predictor}')

    num_pred_cat = list(set([v for vs in cat_num_map.values() for v in vs]))
    num_pred_no_cat = list(set(numerical_predictors).difference(num_pred_cat))

    if active_predictor in numerical_predictors:
        df_counterfactual = DataFrame({active_predictor: np.linspace(
            df[active_predictor].min(), df[active_predictor].max(), COUNTER_FACTUAL_NUM_POINTS)})
    else:
        cat_codes = cat_factors[active_predictor]['cat_feature_codes']

        df_counterfactual = DataFrame({active_predictor: cat_codes})

    for p in numerical_predictors:
        if p != active_predictor:
            df_counterfactual[p] = None

    for p in categorical_predictors:
        if p != active_predictor:
            df_counterfactual[p] = cat_factors[p]['cat_feature_codes_map'][categorical_predictors_vals[p]]
            # df_counterfactual[p] = np.ma.masked_values([-999 for i in range(len(df_counterfactual))], value=-999)

    with model:
        for p in categorical_predictors:
            pm.set_data({f'data_cat[{p}]': df_counterfactual[p].values})

        for p in num_pred_cat:
            pm.set_data({f'data_nc[{p}]': df_counterfactual[p].values})

        if len(num_pred_no_cat) > 0:
            pm.set_data(
                {"data_num": df_counterfactual[num_pred_no_cat].values})

        # use the updated values and predict outcomes and probabilities:
        # thinned_idata = idata.sel(draw=slice(None, None, 5))

        idata_2 = pm.sample_posterior_predictive(
            idata,
            var_names=[target, f'mu_{target}'],
            return_inferencedata=True,
            predictions=True,
        )

    return idata_2, df_counterfactual[active_predictor]


def summarize_predictions_old(idata, predictor, target, predictor_vals, cat_codes):
    predictions = az.extract(idata, 'predictions')

    target_pred = predictions[target]
    mu_pred = predictions[f'mu_{target}']

    target_hdi = az.hdi(idata.predictions)[target]
    mu_hdi = az.hdi(idata.predictions)[f'mu_{target}']
    mu_mean = az.extract(idata.predictions).mean('sample')[f'mu_{target}']

    ps = PredictionSummary(predictor, target_pred, mu_pred,
                           target_hdi, mu_hdi, mu_mean, predictor_vals, cat_codes)

    return ps


def factorize(vals):
    cat_feature_idx, cat_feature_codes = pd.factorize(vals, sort=True)
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
        
    model, idata = sample_model(complete_model)
    # return model, idata

    res = calc_counterfactual_predictor(df, model, idata, graph, topo_order, cat_num_map_per_target)
    return res #, model, idata



def init_complete_model(df,graph):

    with pm.Model() as model:
        coords = {}

        for name,node in graph.nodes.items():
            if node.info.featureType in set([FeatureType.BOOL, FeatureType.CATEGORICAL]):
                coords[name] = node.info.cat_feature_codes
                vals = node.info.cat_feature_idx
            else:
                vals = df[name].values

            v = pm.MutableData(f'data_{name}', vals)

        model.add_coords(coords)

    return model


def create_cat_num_map(df, node, parent_nodes):
    categorical_nodes = [n for n in parent_nodes.values(
    ) if n.info.featureType in set([FeatureType.BOOL, FeatureType.CATEGORICAL])]
    numerical_nodes = [n for n in parent_nodes.values(
    ) if n.info.featureType not in set([FeatureType.BOOL, FeatureType.CATEGORICAL])]

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

def add_sub_model(df, target_node, predictor_nodes, cat_num_map,model):
    target = target_node.name

    categorical_predictors = [name for name, n in predictor_nodes.items(
    ) if n.info.featureType in set([FeatureType.BOOL, FeatureType.CATEGORICAL])]
    numerical_predictors = [name for name, n in predictor_nodes.items(
    ) if n.info.featureType not in set([FeatureType.BOOL, FeatureType.CATEGORICAL])]

    cat_pred_no_num = list(set(categorical_predictors).difference(
        set([p for p, nps in cat_num_map.items() if len(nps) > 0])))
    num_pred_cat = list(set([v for vs in cat_num_map.values() for v in vs]))
    num_pred_no_cat = list(set(numerical_predictors).difference(num_pred_cat))

    print(f'cat_pred_no_num {cat_pred_no_num}')
    print(f'num_pred_no_cat {num_pred_no_cat}')

    with model:

        a = pm.Normal(f'a_{target}', mu=0, sigma=1)
        sigma = pm.Uniform(f'sigma_{target}', 0, 20)

        b_global = {f'bglob_{target}[{p}]': pm.Normal(
            f'bglob_{target}[{p}]', mu=0, sigma=1) for p in categorical_predictors}

        # per-group trend
        bcn = {p: {nmp: pm.Normal(f'bcn_{target}[{p}][{nmp}]', mu=b_global[f'bglob_{target}[{p}]'], sigma=1, dims=p)
                   for nmp in cat_num_map[p]}
               for p in categorical_predictors if p in cat_num_map}

        bcat = {p: pm.Normal(
            f'bcat_{target}[{p}]', mu=b_global[f'bglob_{target}[{p}]'], sigma=1, dims=p) for p in cat_pred_no_num}

        bnum = {p: pm.Normal(
            f'bnum_{target}[{p}]', mu=0, sigma=1) for p in num_pred_no_cat}

        all_trends_cn = sum([sum([bcn[p][nmp][model.named_vars[f'data_{p}']]*model.named_vars[f'data_{nmp}'] for nmp in cat_num_map[p]])
                          for p in categorical_predictors if p in cat_num_map])

        all_trends_c = sum([bcat[p][model.named_vars[f'data_{p}']] for p in cat_pred_no_num])

        all_trends_n = sum([bnum[p]*model.named_vars[f'data_{p}'] for p in num_pred_no_cat])


        mu = pm.Deterministic(
            f'mu_{target}', a+all_trends_cn+all_trends_c+all_trends_n)

        target_var = pm.Normal(target, mu=mu, sigma=sigma,
                               observed=model.named_vars[f'data_{target}'], shape=mu.shape)


def split_categorical_numerical_vars(graph,vars):
    categorical_vars = set([v for v in vars if graph.nodes[v].info.featureType in set([FeatureType.BOOL, FeatureType.CATEGORICAL])])
    numerical_vars = set([v for v in vars if graph.nodes[v].info.featureType not in set([FeatureType.BOOL, FeatureType.CATEGORICAL])])

    return categorical_vars,numerical_vars

#need to block only upstream nodes from the predictor. donwstream predictors should be opne - therefore : None
# block: if numeric=0 else need to set every combination.
def calc_counterfactual_predictor(df, model, idata, graph, topo_order, cat_num_map_per_target):

    res = {}

    for target in topo_order:
        res[target] = {}       
        for active_predictor in graph.nodes[target].parent_vars:

            cat_num_map = cat_num_map_per_target[target]

            #find the downstream vars
            downstream_vars = graph.get_downstream_vars(active_predictor)

            non_blocking_predictors = set(graph.nodes[target].parent_vars).intersection(downstream_vars)
            blocking_predictors = set(graph.nodes[target].parent_vars).difference(non_blocking_predictors)

            categorical_predictors,numerical_predictors = split_categorical_numerical_vars(graph,graph.nodes[target].parent_vars)
            
            idata_2, predictor_vals = process_couterfactual_predictor(df, model, idata, graph, target, active_predictor,
                                cat_num_map,non_blocking_predictors,blocking_predictors,categorical_predictors,numerical_predictors)

            res[target][active_predictor] = {}
            res[target][active_predictor]['idata'] = idata_2
            res[target][active_predictor]['predictor_vals'] = predictor_vals
    
    return res
            

def process_couterfactual_predictor(df, model, idata, graph, target, active_predictor,
                                cat_num_map,non_blocking_predictors,blocking_predictors,categorical_predictors,numerical_predictors):

    print(
        f'process_couterfactual_predictor: target = {target}')
    print(
        f'process_couterfactual_predictor: active_predictor = {active_predictor}')
    print(
        f'process_couterfactual_predictor: cat_num_map = {cat_num_map}')
    print(
        f'process_couterfactual_predictor: non_blocking_predictors = {non_blocking_predictors}')
    print(
        f'process_couterfactual_predictor: blocking_predictors = {blocking_predictors}')
    print(
        f'process_couterfactual_predictor: categorical_predictors = {categorical_predictors}')
    print(
        f'process_couterfactual_predictor: numerical_predictors = {numerical_predictors}')


    if active_predictor in numerical_predictors:
        df_counterfactual = DataFrame({active_predictor: np.linspace(
            df[active_predictor].min(), df[active_predictor].max(), COUNTER_FACTUAL_NUM_POINTS)})
    else:
        cat_codes = cat_factors[active_predictor]['cat_feature_codes']

        df_counterfactual = DataFrame({active_predictor: cat_codes})

    for n,node in graph.nodes.items():
        if n == active_predictor:
            continue

        if n in blocking_predictors:
            #GUYGUY: need to change this. take categorical value from input 
            df_counterfactual[n] = 0 if n in numerical_predictors else graph.nodes[n].info.cat_feature_codes[0]
        else:
            df_counterfactual[n] = np.nan 

    print('df_counterfactual')
    print(df_counterfactual)

    with model:
        for n,node in graph.nodes.items():
            pm.set_data({f'data_{n}': df_counterfactual[n].values})

        # use the updated values and predict outcomes and probabilities:
        # thinned_idata = idata.sel(draw=slice(None, None, 5))

        idata_2 = pm.sample_posterior_predictive(
            idata,
            var_names=[target, f'mu_{target}'],
            return_inferencedata=True,
            predictions=True,
        )

    return idata_2, df_counterfactual[active_predictor]


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
        # idata = jax.sample_numpyro_nuts() # faster
        # idata = jax.sample_blackjax_nuts() # not working

        pm.sample_posterior_predictive(idata, extend_inferencedata=True)

    # post = az.extract(idata)
    # pm.model_to_graphviz(model)
    return model, idata


def create_model_old(df, target, predictors):

    features = predictors + [target]
    col_types = identify_cols(df, features)

    categorical_features = [col for col, col_type in col_types.items()
                            if col_type in set([FeatureType.BOOL, FeatureType.CATEGORICAL])]

    numerical_features = [
        f for f in features if col_types[f] == FeatureType.NUMERICAL]

    print(f'create_model features = {features}')
    print(f'create_model numerical_features = {numerical_features}')

    cat_num_map = {'adult': ['age']}

    return create_model_and_predictions(df, target, predictors, categorical_features,numerical_features,cat_num_map)


def create_model_and_predictions_old(df, target, predictors, categorical_features,numerical_features,cat_num_map):

    categorical_predictors = [
        col for col in categorical_features if col in predictors]
    numerical_predictors = [
        col for col in numerical_features if col in predictors]

    feature_means = {f: df[f].mean() for f in numerical_features}
    feature_stds = {f: df[f].std() for f in numerical_features}

    for f in numerical_features:
        df[f] = standardize_vec(df[f], feature_means[f], feature_stds[f])

    cat_factors = {col: factorize(df[col]) for col in categorical_predictors}

    model, idata = create_num_to_features_model(df, target, categorical_predictors,numerical_predictors,cat_num_map,cat_factors)

    categorical_predictors_vals = {
        p: cat_factor['cat_feature_codes'][0] for p, cat_factor in cat_factors.items()}

    # return model,idata,categorical_predictors,numerical_predictors,feature_means,feature_stds,cat_factors,categorical_predictors_vals

    def unstand(predictor, ps):
        if predictor in feature_means:
            ps.predictor_vals = unstandardize_vec(
                ps.predictor_vals, feature_means[predictor], feature_stds[predictor])
        if target in feature_means:
            ps.target_hdi = unstandardize_vec(
                ps.target_hdi, feature_means[target], feature_stds[target])
            ps.mu_hdi = unstandardize_vec(
                ps.mu_hdi, feature_means[target], feature_stds[target])
            ps.mu_mean = unstandardize_vec(
                ps.mu_mean, feature_means[target], feature_stds[target])
            ps.target_pred = unstandardize_vec(
                ps.target_pred, feature_means[target], feature_stds[target])
            ps.mu_pred = unstandardize_vec(
                ps.mu_pred, feature_means[target], feature_stds[target])

    def smooth(ps):
        ps.target_hdi = savgol_filter(
            ps.target_hdi, axis=0, window_length=COUNTER_FACTUAL_SMOOTH_NUM_POINTS, polyorder=2)
        ps.mu_hdi = savgol_filter(
            ps.mu_hdi, axis=0, window_length=COUNTER_FACTUAL_SMOOTH_NUM_POINTS, polyorder=2)
        ps.mu_mean = savgol_filter(
            ps.mu_mean, axis=0, window_length=COUNTER_FACTUAL_SMOOTH_NUM_POINTS, polyorder=2)

    def calc_counterfactual(active_predictor):
        idata_2, predictor_vals = calc_counterfactual_predictor(df, model, idata,target,active_predictor,
                                                                categorical_predictors, numerical_predictors,
                                                                cat_num_map, categorical_predictors_vals, cat_factors)
        cat_codes = cat_factors[active_predictor]['cat_feature_codes'] if active_predictor in cat_factors else [
        ]

        ps = summarize_predictions(idata_2, active_predictor, target, predictor_vals,cat_codes)

        return ps

    pred_res = {active_predictor: calc_counterfactual(active_predictor)
                for active_predictor in predictors}

    for predictor, ps in pred_res.items():
        unstand(predictor, ps)
        if predictor in numerical_predictors:
            smooth(ps)

    return pred_res
