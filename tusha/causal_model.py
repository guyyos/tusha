import dash                     # pip install dash
from dash.dependencies import Input, Output, State
import dash_core_components as dcc
import dash_html_components as html
import plotly.express as px     # pip install plotly==5.2.2
from dash import dcc, html, callback, Output, Input
import dash_bootstrap_components as dbc

import pandas as pd
from pandas import DataFrame
import uuid
import time
from app import cache
import datetime
import dash_table
import numpy as np
from dash import Dash, dcc, html, Input, Output, State, ctx, MATCH, ALL
import json
from model_creation import create_categorical_univariate_model,create_numerical_univariates_model,create_numerical_univariate_model,create_cat_to_num_model,create_num_to_num_model
from multivar_model_creation import create_complete_model
from plot_creation import create_plots_with_reg_hdi_lines
import arviz as az
import plotly.figure_factory as ff
import dash_cytoscape as cyto
from multivar_model_creation_time import create_complete_time_model

causal_model_layout = html.Div(id='causal_model_layout')

# return html.Div(
#     [
#         dbc.Button(className="bi bi-trash  rounded-circle m-4", outline=True, color="primary"),
#         dbc.Button(className="bi bi-plus-lg rounded-circle", outline=True, color="primary")
#     ])

def get_causal_model_layout(df):

    df1 = DataFrame(columns = ['Cause','Effect'])

    causal_layout = dbc.Container([
        dcc.Store(data={}, id='causal-graph'),
        # dbc.Row(
        #     [
        #         dbc.Col([dbc.Label("feature:"),
        #                  dcc.Dropdown(id='chosen_univar_plot', value='',options=[{'label': x, 'value': x} for x in df.columns]),dbc.Container([],id='univar_process_spinner')], width=2),
        #         dbc.Col(dbc.Container(id='univar_plot'), align="center",)]),
        # html.Br(),
        dbc.Row([
            dbc.Col([dbc.Label("Cause:"), 
            dcc.Dropdown(id='new_cause', value='',options=[{'label': x, 'value': x} for x in df.columns])], 
                                                       width=2,align="center"),

            dbc.Col([dbc.Label("Effect:"), 
            dcc.Dropdown(id='new_effect', value='',options=[],disabled=True)],
                                                        width=2,align="center"),
            dbc.Col(
                    dbc.Button(id="add-cause-effect", className="bi bi-plus-lg rounded-circle",
                               outline=True, color="primary", n_clicks=0,disabled=True),
                               width=2,align="end")
            ],justify="right"
        ),
        dbc.Tooltip("Add cause effect relation",
                            target="add-cause-effect"),
        dbc.Toast(
            header = "Cause-effect relation already exists",
            id="cause-effect-exists-msg",
            icon="warning",
            duration=4000,
            is_open=False,
        ),

        html.Hr(),
        dbc.Row([
        dbc.Col(dash_table.DataTable(
            id='cause-effect-relations',
            columns=[
                {"name": i, "id": i, "deletable": False, "selectable": False} for i in df1.columns
            ],
            data=df1.to_dict('records'),
            editable=True,
            filter_action="native",
            sort_action="native",
            sort_mode="multi",
            column_selectable="single",
            row_selectable="multi",
            row_deletable=True,
            selected_columns=[],
            selected_rows=[],
            page_action="native",
            page_current=0,
            page_size=10,
            style_cell={
                            # 'padding-right': '30px',
                            # 'padding-left': '10px',
                            'text-align': 'center',
                            'marginLeft': 'auto',
                            'marginRight': 'auto'
                        }
        ),width=5),
        dbc.Col(dbc.Container(id='causal-net', children=[]),width=6)]),
        html.Hr(),
        dbc.Row([dbc.Col([dbc.Button(id="build-model-button",
                                    outline=True, color="primary",
                n_clicks=0, className='bi bi-hammer rounded-circle'),
                dbc.Container(id='build_model_spinner', children=[])], width=1),
                dbc.Col(dbc.Button(id="cancel-build",outline=True, color="primary",
                                   n_clicks=0, className='bi bi-x-lg rounded-circle'), width=1)
                 ]),
        dbc.Tooltip("Build model",
                            target="build-model-button"),
        dbc.Container(id='model-res', children=[])
    ])

    return causal_layout

def get_all_parent_causes(effect_causes,feature):

    def get_all_parent_causes_recurs(effect_causes,feature,causes):
        if feature in causes:
            return
        causes.add(feature)
        for cause in effect_causes[feature]:
            get_all_parent_causes_recurs(effect_causes,cause,causes)

    causes = set([])
    get_all_parent_causes_recurs(effect_causes,feature,causes)

    return causes

def find_possible_effects(session_id,df_relations,new_cause):
    df = query_data(session_id)
    all_features = df.columns

    effect_causes = df_relations.groupby('Effect')['Cause'].apply(list)

    #add features that were not added yet
    for f in all_features:
        if f not in effect_causes:
            effect_causes[f] = []

    causes = get_all_parent_causes(effect_causes,new_cause)
    posibble_effects = [f for f in all_features if f not in causes]

    return posibble_effects


def remove_existing_relations(df_relations,new_cause,posibble_effects):
    return [effect for effect in posibble_effects if \
            len(df_relations)<=0 or \
            len(df_relations[(df_relations['Cause']==new_cause)&(df_relations['Effect']==effect)])<=0]

@callback(
    Output('new_effect', 'options'),
    Output('new_effect', 'value'),
    Output('new_effect', 'disabled'),
    Input('new_cause', 'value'),
    State('cause-effect-relations', 'data'),
    State('session-id', 'data')
)
def populate_effect(new_cause, cause_effect_rels,session_id):
    if new_cause is None or new_cause=='':
        return [],'',True
    
    df_relations = DataFrame(columns = ['Cause','Effect'],data=cause_effect_rels)

    posibble_effects = find_possible_effects(session_id,df_relations,new_cause)
    posibble_effects = remove_existing_relations(df_relations,new_cause,posibble_effects)
    
    deafult_effect = posibble_effects[0] if len(posibble_effects)>0 else None
    return posibble_effects,deafult_effect,False


@callback(
    Output('add-cause-effect', 'disabled'),
    Input('new_effect', 'value'),
    State('session-id', 'data')
)
def enable_add_relation(new_effect, session_id):
    if new_effect is None or new_effect=='':
        return True

    return False


@callback(
    Output('cause-effect-relations', 'data'),
    Output("cause-effect-exists-msg", "is_open"),
    Output('causal-net','children'),
    Input('add-cause-effect', 'n_clicks'),
    [State('new_cause', 'value'),
     State('new_effect', 'value'),
    State('cause-effect-relations', 'data'),
    State('session-id', 'data'),
    State('causal-net','children')]
)
def add_cause_effect(add_ce_clicks, new_cause,new_effect, cause_effect_rels,session_id,cur_causal_net):

    if add_ce_clicks <=0:
        return cause_effect_rels,False,cur_causal_net
    
    df_relations = DataFrame(columns = ['Cause','Effect'],data=cause_effect_rels)
    if len(df_relations[(df_relations['Cause']==new_cause)&(df_relations['Effect']==new_effect)])>0:
        return cause_effect_rels,True,cur_causal_net

    posibble_effects = find_possible_effects(session_id,df_relations,new_cause)

    if new_effect in posibble_effects:
        cause_effect_rels.append({'Cause': new_cause, 'Effect': new_effect})

    df_relations1 = pd.concat([df_relations,DataFrame([{"Cause": new_cause,"Effect": new_effect}])])
    print(df_relations1)
    causal_net = generate_causal_net(df_relations1)

    return cause_effect_rels,False,[causal_net]

def generate_causal_net(df_relations):
    directed_edges = df_relations.apply(lambda x:{'data': {'id': x['Cause']+x['Effect'], 'source': x['Cause'], 'target': x['Effect']}},axis=1)

    nodes = set(df_relations.Cause.unique())|set(df_relations.Effect.unique())

    directed_elements = [{'data': {'id': id_}} for id_ in nodes] + list(directed_edges)


    net = cyto.Cytoscape(
            id='net1',
            layout={'name': 'breadthfirst','animate': True},#cose  breadthfirst
            style={'height': '250px'},#'width': '40%'

            stylesheet=[
        
                {
                    'selector': 'node',
                    'style': {
                        'label': 'data(id)'
                    }
                },
                {
                    'selector': 'edge',
                    'style': {
                        'curve-style': 'bezier',
                        'target-arrow-color': 'blue',
                        'target-arrow-shape': 'vee',
                        'line-color': 'blue'
                    }
                }
            ],
            elements=directed_elements
        )
    
    return net




# @cache.memoize()
def query_data(session_id):
    print(f'query_data {session_id}')
    data = cache.get(session_id+'_data')
    print(f'query_data data: {str(data)[:100]}...{str(data)[-100:]}')

    return pd.read_json(data)


@dash.callback(Output('model-res', 'children'),
               Input('build-model-button', 'n_clicks'),
               [State('session-id', 'data'),
                State('cause-effect-relations', 'data'),
                State('time_col', "data")],
               background=True,
               running=[
    (Output("build-model-button", "disabled"), True, False),
    (Output("cancel-build", "disabled"), False, True),
    (Output("build_model_spinner","children"),[dbc.Spinner(size="sm")],[])
    ],
    cancel=[Input("cancel-build", "n_clicks")]
)
def build_model(n_clicks, session_id,cause_effect_rels,time_col):
    if n_clicks is None or n_clicks<=0:
        return None

    df_relations = DataFrame(columns = ['Cause','Effect'],data=cause_effect_rels)
    figs = []

    # for index, row in df_relations.iterrows():
    #     feature,target = row['Cause'], row['Effect']
    #     print(f'{index} {feature},{target}')
    #     fig = get_bivariate_plot(session_id,target,feature)
    #     fig_name = f'{feature}→{target}'
    #     fig.update_layout(title=fig_name) 
    #     figs.append(dcc.Graph(id = fig_name,figure=fig))

    figs2 = get_model_plots(session_id,df_relations,time_col)

    figs+=figs2
    
    return figs


@cache.memoize()
def get_model_plots(session_id,df_relations,time_col):
    df = pd.read_json(cache.get(session_id+'_data'))
    df = df.dropna()
    all_figs = []

    if time_col:
        df_temporal,model,res,summary_res,graph = create_complete_time_model(df.copy(),df_relations,time_col)
        df = df_temporal
    else:
        model,res,summary_res,graph = create_complete_model(df.copy(),df_relations)

    figs =  create_plots_with_reg_hdi_lines(df,summary_res,graph)

    for target,target_fig in figs.items():

        for predictor,pred_fig in target_fig.items():
            fig = pred_fig['fig']
            predictors = pred_fig['predictors']

            preds = ','.join([f'<b>{p}</b>' if p==predictor else p for p in predictors])
            fig_name = f'[{preds}]<b>→{target}</b>'
            fig.update_layout(title=fig_name) 
            all_figs.append(dcc.Graph(id = fig_name,figure=fig))

            if pred_fig['fig_mu']:
                fig = pred_fig['fig_mu']
                fig_name = f'mu: [{predictor}]<b>→{target}</b>'
                fig.update_layout(title=fig_name) 
                all_figs.append(dcc.Graph(id = fig_name,figure=fig))

    return all_figs


@cache.memoize()
def get_bivariate_plot(session_id,target,feature):
    df = pd.read_json(cache.get(session_id+'_data'))
    df = df.dropna()
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    if feature in df.select_dtypes(include=numerics).columns:
        fig = create_num_to_num_model(df,target,feature)
    else:
        res = create_cat_to_num_model(df,target,feature)
        group_labels = list(res.keys())
        fig = ff.create_distplot(list(res.values()),group_labels, show_hist=False,show_rug=False)


    return fig


@cache.memoize()
def get_numerical_univariate_plot(df,target):

    res = create_numerical_univariate_model(df,target)
    print(f'get_univariate_plots idata = {res}')

    group_labels = list(res.keys())
    fig = ff.create_distplot(list(res.values()),group_labels, show_hist=False,show_rug=False)

    return fig

@cache.memoize()
def get_cat_univariate_plot(df,target):

    res = create_categorical_univariate_model(df,target)
    print(f'get_cat_univariate_plots res = {res}')

    group_labels = list(res.keys())
    fig = ff.create_distplot(list(res.values()),group_labels, show_hist=False,show_rug=False)

    return fig


@cache.memoize()
def get_univariate_plot(session_id,target):
    df = pd.read_json(cache.get(session_id+'_data'))
    df = df.dropna()

    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    if target in df.select_dtypes(include=numerics).columns:
        fig = get_numerical_univariate_plot(df,target)
    else:
        fig = get_cat_univariate_plot(df,target)

    return fig
    


# @dash.callback(Output('univar_plot', 'children'),
#                Input('chosen_univar_plot', 'value'),
#                State('session-id', 'data'),
#                background=True,
#                running=[(Output("chosen_univar_plot", "disabled"), True, False),
#                         (Output("univar_process_spinner","children"),[dbc.Spinner(size="sm")],[])])
# def plot_univariate(chosen_univar_plot, session_id):
#     if chosen_univar_plot is None or chosen_univar_plot=='':
#         return None

#     print(f'plot_univariate {session_id} {chosen_univar_plot}')
#     fig = get_univariate_plot(session_id,chosen_univar_plot)

#     return [dcc.Graph(figure=fig)]



# def get_numeric_univariate_plot(df):

#     numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
#     idata = create_numerical_univariates_model(df.dropna().select_dtypes(include=numerics))
#     print(f'get_univariate_plots idata = {idata}')
#     post = az.extract(idata)

#     print('process_data:')
#     print(post)

#     # fig = px.histogram(tips, x="total_bill", y="tip", color="sex", marginal="rug",
#     #                hover_data=tips.columns)
#     # fig = px.histogram(DataFrame(columns = [param_name],data=az.extract(post)[param_name].values), x=param_name)

#     children = []
#     cols = []

#     for v in post:
#         group_labels = [v]

#         fig = ff.create_distplot([post[v].values],group_labels, show_hist=False,show_rug=False)
#         fig.update_layout(height=300)
#         fig.update_layout(legend={'entrywidth':0.1,'entrywidthmode':'fraction'})

#         cols.append(dbc.Col(dcc.Graph(figure=fig),width=4))
#         if len(cols)>=2:
#             children.append(dbc.Row(cols))
#             cols = []

#     if len(cols)>0:
#         children.append(dbc.Row(cols))

#     return children

# def get_cat_univariate_plots(df):
#     import plotly.figure_factory as ff

#     numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
#     df = df.dropna().select_dtypes(exclude=numerics)
#     children = []

#     for col in df.columns:
#         res = create_categorical_univariate_model(df,col)
#         print(f'get_cat_univariate_plots res = {res}')

#         group_labels = list(res.keys())

#         fig = ff.create_distplot(list(res.values()),group_labels, show_hist=False,show_rug=False)

#         children.append(dbc.Row(dbc.Col(dcc.Graph(figure=fig),width=4)))

#     return children


