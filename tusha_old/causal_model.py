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
from model_creation import create_model_example1
import arviz as az
import plotly.figure_factory as ff
import dash_cytoscape as cyto


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
        html.Br(),
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
        dbc.Row([dbc.Col(dbc.Button(id="build-model-button",
                                    outline=True, color="primary",
                n_clicks=0, className='bi bi-hammer rounded-circle'), width=1),
                dbc.Col(dbc.Button(id="cancel-build",outline=True, color="primary",
                                   n_clicks=0, className='bi bi-x-lg rounded-circle'), width=1)
                 ]),
        dbc.Tooltip("Build model",
                            target="build-model-button"),
        dbc.Container(id='model-res', children=[])
    ])

    return causal_layout

def get_all_parent_causes(causal_graph,feature):

    def get_all_parent_causes_recurs(causal_graph,feature,causes):
        if feature in causes:
            return
        causes.add(feature)
        for cause in causal_graph[feature]:
            get_all_parent_causes_recurs(causal_graph,cause,causes)

    causes = set([])
    get_all_parent_causes_recurs(causal_graph,feature,causes)

    return causes

def find_possible_effects(session_id,df_relations,new_cause):
    df = query_data(session_id)
    all_features = df.columns

    causal_graph = df_relations.groupby('Effect')['Cause'].apply(list)

    #add features that were not added yet
    for f in all_features:
        if f not in causal_graph:
            causal_graph[f] = []

    causes = get_all_parent_causes(causal_graph,new_cause)
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




@cache.memoize()
def query_data(session_id):
    return pd.read_json(cache.get(session_id+'_data'))


@dash.callback(Output('model-res', 'children'),
               Input('build-model-button', 'n_clicks'),
               [State('session-id', 'data'),
                State('cause-effect-relations', 'data')],
               background=True,
               running=[
    (Output("build-model-button", "disabled"), True, False),
    (Output("cancel-build", "disabled"), False, True),
],
    cancel=[Input("cancel-build", "n_clicks")]
)
def process_data(n_clicks, session_id,rows):
    if n_clicks is None or n_clicks<=0:
        return None
    return temp_plot()


    idata = create_model_example1()
    post = idata.posterior

    print('process_data:')
    print(post)

    # fig = px.histogram(tips, x="total_bill", y="tip", color="sex", marginal="rug",
    #                hover_data=tips.columns)
    fig = px.histogram(DataFrame(columns = ['mu'],data=az.extract(post).mu.values), x="mu")

    print(' $$$$$$$$$$$$$  after create_distplot')


    return [dbc.Row([dbc.Col(dcc.Graph(figure=fig),width=4),dbc.Col(dcc.Graph(figure=fig),width=4)])]

def temp_plot():
    import plotly.graph_objects as go

    fig = go.Figure(go.Sankey(
        arrangement='snap',
        node=dict(
            label=['A', 'B', 'C', 'D', 'E', 'F'],
            x=[0.2, 0.1, 0.5, 0.7, 0.3, 0.5],
            y=[0.7, 0.5, 0.2, 0.4, 0.2, 0.3],
            pad=10  
        ),
        link=dict(
            source=[0, 0, 1, 2, 5, 4, 3, 5],
            target=[5, 3, 4, 3, 0, 2, 2, 3],
            value=[1, 2, 1, 1, 1, 1, 1, 2]  
        )
    ))

    return [dcc.Graph(figure=fig)]


# @dash.callback(Output('model-res', 'children'),
#                Input('build-model-button', 'n_clicks'),
#                [State('session-id', 'data'),
#                 State('cause-effect-relations', 'data')],
#                background=True,
#                running=[
#     (Output("build-model-button", "disabled"), True, False),
#     (Output("cancel-build", "disabled"), False, True),
# ],
#     cancel=[Input("cancel-build", "n_clicks")]
# )
# def process_data(n_clicks, session_id,rows):
#     if n_clicks is None:
#         return None

#     dd = rows
#     return html.Pre(dd)
