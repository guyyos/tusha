import os
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
from model_creation import create_categorical_univariate_model, create_numerical_univariates_model, create_numerical_univariate_model, create_cat_to_num_model, create_num_to_num_model
from multivar_model_creation import create_complete_model
import arviz as az
import plotly.figure_factory as ff
import dash_cytoscape as cyto
from multivar_model_creation_time import create_complete_time_model
from load_data_tab import load_data
import graphviz
import base64
from file_handle import clean_user_model,get_full_name,save_file


def get_initial_layout():

    df = DataFrame()
    df1 = DataFrame(columns=['Cause', 'Effect'])

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
                     dcc.Dropdown(id='new_cause', value='', options=[{'label': x, 'value': x} for x in df.columns])],
                    width=2, align="center"),

            dbc.Col([dbc.Label("Effect:"),
                     dcc.Dropdown(id='new_effect', value='', options=[], disabled=True)],
                    width=2, align="center"),
            dbc.Col(
                dbc.Button(id="add-cause-effect", className="bi bi-plus-lg rounded-circle",
                           outline=True, color="primary", n_clicks=0, disabled=True),
                width=2, align="end")
        ], justify="right"
        ),
        dbc.Tooltip("Add cause effect relation",
                    target="add-cause-effect"),
        dbc.Toast(
            header="Cause-effect relation already exists",
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
            ), width=5),
            dbc.Col(dbc.Container(id='causal-net', children=[]), width=6)]),
        html.Hr(),
        dbc.Row([dbc.Col(dbc.Button(id="build-model-button",
                                    outline=True, color="primary",
                                    n_clicks=0, className='bi bi-hammer rounded-circle'), width=1),

                dbc.Col(dbc.Button(id="cancel-build", outline=True, color="primary",
                                   n_clicks=0, className='bi bi-x-lg rounded-circle'), width=1),

                dbc.Col(dbc.Container(
                    id='build_model_spinner', children=[]), width=1)
                 ]),
        dbc.Row(dbc.Toast(
            header="",
            id="build-model-err-msg",
            icon="warning",
            duration=4000,
            is_open=False,
        )),
        dbc.Tooltip("Build model",
                    target="build-model-button"),

        html.Hr(),
        dbc.Container(id='model-plate', children=[])])

    return causal_layout



causal_model_layout = html.Div(
    id='causal_model_layout', children=get_initial_layout())


@callback(Output('new_cause', 'options'),
          Input('prev_file_selector', 'value'),
          State('session-id', 'data'))
def populate_cause(file_updated, session_id):

    if file_updated:
        df, time_col = load_data(session_id)
        features = list(df.columns)
        if time_col in features: features.remove(time_col)

        return features

    return []


def get_all_parent_causes(effect_causes, feature):

    def get_all_parent_causes_recurs(effect_causes, feature, causes):
        if feature in causes:
            return
        causes.add(feature)
        for cause in effect_causes[feature]:
            get_all_parent_causes_recurs(effect_causes, cause, causes)

    causes = set([])
    get_all_parent_causes_recurs(effect_causes, feature, causes)

    return causes


def find_possible_effects(session_id, df_relations, new_cause):
    df, time_col = load_data(session_id)
    all_features = df.columns

    effect_causes = df_relations.groupby('Effect')['Cause'].apply(list)

    # add features that were not added yet
    for f in all_features:
        if f not in effect_causes:
            effect_causes[f] = []

    causes = get_all_parent_causes(effect_causes, new_cause)
    posibble_effects = [f for f in all_features if f not in causes]

    return posibble_effects


def remove_existing_relations(df_relations, new_cause, posibble_effects):
    return [effect for effect in posibble_effects if
            len(df_relations) <= 0 or
            len(df_relations[(df_relations['Cause'] == new_cause) & (df_relations['Effect'] == effect)]) <= 0]


@callback(
    Output('new_effect', 'options'),
    Output('new_effect', 'value'),
    Output('new_effect', 'disabled'),
    Input('new_cause', 'value'),
    State('cause-effect-relations', 'data'),
    State('session-id', 'data')
)
def populate_effect(new_cause, cause_effect_rels, session_id):
    if new_cause is None or new_cause == '':
        return [], '', True

    df_relations = DataFrame(
        columns=['Cause', 'Effect'], data=cause_effect_rels)

    posibble_effects = find_possible_effects(
        session_id, df_relations, new_cause)
    posibble_effects = remove_existing_relations(
        df_relations, new_cause, posibble_effects)

    deafult_effect = posibble_effects[0] if len(posibble_effects) > 0 else None
    return posibble_effects, deafult_effect, False


@callback(
    Output('add-cause-effect', 'disabled'),
    Input('new_effect', 'value'),
    State('session-id', 'data')
)
def enable_add_relation(new_effect, session_id):
    if new_effect is None or new_effect == '':
        return True

    return False


@callback(
    Output('cause-effect-relations', 'data'),
    Output("cause-effect-exists-msg", "is_open"),
    Output('causal-net', 'children'),
    Input('add-cause-effect', 'n_clicks'),
    Input('cause-effect-relations', 'data'),
    [State('new_cause', 'value'),
     State('new_effect', 'value'),
     State('session-id', 'data'),
     State('causal-net', 'children')]
)
def add_cause_effect(add_ce_clicks, cause_effect_rels, new_cause, new_effect, session_id, cur_causal_net):
    if add_ce_clicks <= 0:
        return cause_effect_rels, False, cur_causal_net

    clean_user_model(session_id)

    btn = dash.callback_context.triggered[0]["prop_id"].split(".")[0]
    print(f'add_cause_effect btn = {btn}')

    if btn == "cause-effect-relations":
        df_relations = DataFrame(
            columns=['Cause', 'Effect'], data=cause_effect_rels)
        causal_net = generate_causal_net(df_relations)
        return cause_effect_rels, False, [causal_net]

    df_relations = DataFrame(
        columns=['Cause', 'Effect'], data=cause_effect_rels)
    if len(df_relations[(df_relations['Cause'] == new_cause) & (df_relations['Effect'] == new_effect)]) > 0:
        return cause_effect_rels, True, cur_causal_net

    posibble_effects = find_possible_effects(
        session_id, df_relations, new_cause)

    if new_effect in posibble_effects:
        cause_effect_rels.append({'Cause': new_cause, 'Effect': new_effect})

    df_relations1 = pd.concat([df_relations, DataFrame(
        [{"Cause": new_cause, "Effect": new_effect}])])
    print(df_relations1)
    causal_net = generate_causal_net(df_relations1)

    return cause_effect_rels, False, [causal_net]


def generate_causal_net(df_relations):
    if len(df_relations) <= 0:
        return []

    directed_edges = df_relations.apply(lambda x: {'data': {
                                        'id': x['Cause']+x['Effect'], 'source': x['Cause'], 'target': x['Effect']}}, axis=1)

    nodes = set(df_relations.Cause.unique()) | set(
        df_relations.Effect.unique())

    directed_elements = [{'data': {'id': id_}}
                         for id_ in nodes] + list(directed_edges)

    net = cyto.Cytoscape(
        id='net1',
        layout={'name': 'breadthfirst', 'animate': True},  # cose  breadthfirst
        style={'height': '250px'},  # 'width': '40%'

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


@dash.callback(Output('model-plate', 'children'),
               Output('build-model-err-msg', "is_open"),
               Output('build-model-err-msg', 'header'),
               Input('build-model-button', 'n_clicks'),
               Input('model-plate', 'children'),
               [State('session-id', 'data'),
                State('cause-effect-relations', 'data')],
               background=True,
               running=[
    (Output("build-model-button", "disabled"), True, False),
    (Output("cancel-build", "disabled"), False, True),
    (Output("build_model_spinner", "children"), [dbc.Spinner(size="sm")], [])
],
    cancel=[Input("cancel-build", "n_clicks")]
)
def construct_model(n_clicks, model_plate, session_id, cause_effect_rels):
    if n_clicks <= 0:
        return model_plate, False, ''

    if len(cause_effect_rels) <= 0:
        return model_plate, True, 'Populate cause-effect table'

    btn = dash.callback_context.triggered[0]["prop_id"].split(".")[0]

    df_relations = DataFrame(
        columns=['Cause', 'Effect'], data=cause_effect_rels)
    figs = []

    # for index, row in df_relations.iterrows():
    #     feature,target = row['Cause'], row['Effect']
    #     print(f'{index} {feature},{target}')
    #     fig = get_bivariate_plot(session_id,target,feature)
    #     fig_name = f'{feature}→{target}'
    #     fig.update_layout(title=fig_name)
    #     figs.append(dcc.Graph(id = fig_name,figure=fig))

    df, df1, model, graph, topo_order, cat_num_map_per_target, model_plate = create_model(
        session_id, df_relations)
    save_file('model', session_id, model)
    save_file('graph', session_id, graph)
    save_file('topo_order', session_id, topo_order)
    save_file('cat_num_map_per_target', session_id, cat_num_map_per_target)
    save_file('df', session_id, df)
    save_file('df1', session_id, df1)

    return model_plate, False, ''


# @cache.memoize()
def create_model(session_id, df_relations):
    df, time_col = load_data(session_id)
    df = df.dropna()

    # guyguy
    # if time_col:
    #     df_temporal,model,res,summary_res,graph = create_complete_time_model(df.copy(),df_relations,time_col)
    #     df = df_temporal
    # else:
    #     model,res,summary_res,graph = create_complete_model(df.copy(),df_relations)

    df1, model, graph, topo_order, cat_num_map_per_target, plate_plot = create_complete_model(
        df.copy(), df_relations)

    # Convert Graphviz graph to Plotly figure
    graphviz_graph = graphviz.Source(plate_plot.source)
    svg_str = graphviz_graph.pipe(format='svg').decode('utf-8')

    return df, df1, model, graph, topo_order, cat_num_map_per_target, [dcc.Graph(
        id='example-graph',
        figure={
            'data': [],
            'layout': {
                'height': 800,  # Adjust this value
                'images': [{
                    # 'xref': 'paper',
                    # 'yref': 'paper',
                    'x': 0.5,
                    'y': 0.5,
                    'sizex': 1,
                    'sizey': 1,
                    'xanchor': 'center',
                    'yanchor': 'middle',
                    'layer': 'below',
                    'source': 'data:image/svg+xml;base64,' + base64.b64encode(svg_str.encode('utf-8')).decode('utf-8')
                }],
                'xaxis': {'visible': False},
                'yaxis': {'visible': False}
            }
        }
    )]




@cache.memoize()
def get_bivariate_plot(session_id, target, feature):
    df, time_col = load_data(session_id)
    df = df.dropna()
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    if feature in df.select_dtypes(include=numerics).columns:
        fig = create_num_to_num_model(df, target, feature)
    else:
        res = create_cat_to_num_model(df, target, feature)
        group_labels = list(res.keys())
        fig = ff.create_distplot(
            list(res.values()), group_labels, show_hist=False, show_rug=False)

    return fig


@cache.memoize()
def get_numerical_univariate_plot(df, target):

    res = create_numerical_univariate_model(df, target)
    print(f'get_univariate_plots idata = {res}')

    group_labels = list(res.keys())
    fig = ff.create_distplot(
        list(res.values()), group_labels, show_hist=False, show_rug=False)

    return fig


@cache.memoize()
def get_cat_univariate_plot(df, target):

    res = create_categorical_univariate_model(df, target)
    print(f'get_cat_univariate_plots res = {res}')

    group_labels = list(res.keys())
    fig = ff.create_distplot(
        list(res.values()), group_labels, show_hist=False, show_rug=False)

    return fig


@cache.memoize()
def get_univariate_plot(session_id, target):
    df, time_col = load_data(session_id)
    df = df.dropna()

    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    if target in df.select_dtypes(include=numerics).columns:
        fig = get_numerical_univariate_plot(df, target)
    else:
        fig = get_cat_univariate_plot(df, target)

    return fig

