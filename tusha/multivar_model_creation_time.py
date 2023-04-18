import pymc as pm
import numpy as np
import arviz as az
import pandas as pd
from pandas import DataFrame
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from scipy.signal import savgol_filter
import pytensor.tensor as at
from identify_features import identify_series,find_hierarchical_struct
from base_def import COUNTER_FACTUAL_NUM_POINTS, COUNTER_FACTUAL_SMOOTH_NUM_POINTS, PredictionSummary
from base_def import FeatureInfo, DAGraph, FeatureType
from pymc.sampling import jax
import pytensor.tensor as pt
import xarray as xr
import random
from multivar_model_creation import create_complete_model

NUM_PAST_STEPS = 2


def get_shift_vals(vals,time_past):
    return vals[NUM_PAST_STEPS-time_past:-time_past] if time_past>0 else vals[NUM_PAST_STEPS:]


def shift_df_to_past(df,time_col,cols_to_shift, hierarchy_cols,num_past_steps):
    time_series_level_col = hierarchy_cols[-1] if hierarchy_cols else None

    grs = []
    causes_past_map = {}
    for gr_name,gr in df.groupby(time_series_level_col):
        gr = gr.sort_values(time_col).copy()

        for col in cols_to_shift:
            causes_past_map[col] = {}
            for i in range(1,num_past_steps+1):
                causes_past_map[col][i] = f'{col}[previous_time-{i}]'
                gr[causes_past_map[col][i]] = gr[[col]].shift(periods=i, fill_value=0)

        gr = gr.iloc[num_past_steps:]
        grs.append(gr)

    df_shifted = pd.concat(grs)

    return df_shifted,causes_past_map


def create_complete_time_model(df, df_relations,time_col):
    hierarchy_cols = find_hierarchical_struct(df,time_col)

    causes = set(df_relations.Cause.unique())
    effects = set(df_relations.Effect.unique())
    features_original = causes|effects

    df_temporal,causes_past_map = shift_df_to_past(df,time_col,causes, hierarchy_cols,NUM_PAST_STEPS)

    causes_past = set([cp for cpi in causes_past_map.values() for cp in cpi.values()])

    relations_original = df_relations.apply(lambda x: (x['Cause'], x['Effect']), axis=1)
    relations_no_hierarchy = [(cause,effect)  for (cause,effect) in relations_original if cause not in hierarchy_cols]
    relations_temporal = [ (cp,effect) for (cause,effect) in relations_no_hierarchy for cp in causes_past_map[cause].values()]

    relations_hierarchy = [(h1,h2) for (h1,h2) in zip(hierarchy_cols[:-1],hierarchy_cols[1:])]

    df_relations_new = DataFrame([{'Cause':h1,'Effect':h2} for (h1,h2) in relations_hierarchy+relations_temporal])
    print(f'create_complete_time_model: {df_relations_new}')

    model,res,summary_res,graph = create_complete_model(df_temporal.copy(), df_relations_new)

    return df_temporal,model,res,summary_res,graph


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


