import dash                     # pip install dash
from dash.dependencies import Input, Output, State
import dash_core_components as dcc
import dash_html_components as html
import plotly.express as px     # pip install plotly==5.2.2
from dash import dcc, html, callback, Output, Input
import dash_bootstrap_components as dbc

import pandas as pd
import uuid
import time
from app import cache
import datetime
import dash_table
import numpy as np
from dash import Dash, dcc, html, Input, Output, State, ctx, MATCH, ALL
import json

causal_model_layout = html.Div(id='causal_model_layout')

# return html.Div(
#     [
#         dbc.Button(className="bi bi-trash  rounded-circle m-4", outline=True, color="primary"),
#         dbc.Button(className="bi bi-plus-lg rounded-circle", outline=True, color="primary")
#     ])
def get_causal_model_layout(df):

    causal_layout = dbc.Container([
        html.Br(),
        dbc.Container(id='cause-effect-relations', children=[]),

        dbc.Row(
            [
                dbc.Row(dbc.Col(
        dbc.Button(id="add-cause-effect",className="bi bi-plus-lg rounded-circle", 
                   outline=True, color="primary",n_clicks=0),
               width=2)),
               dbc.Tooltip("Add cause effect relation",target="add-cause-effect")

            ], align="center",),
        html.Hr(),
        dbc.Row([dbc.Col(dbc.Button('Build model', id="build-model-button",
                n_clicks=0, className='rounded-pill'), width=2),
                dbc.Col(dbc.Button('Cancel', id="cancel-build",
                n_clicks=0, className='rounded-pill'), width=2)
                ]),
        dbc.Container(id='test1', children=[]),
        dbc.Container(id='model-res', children=[])

    ])

    return causal_layout


@callback(
    Output('cause-effect-relations', 'children'),
    Input('add-cause-effect', 'n_clicks'),
    Input({'type': 'remove_ce', 'index': ALL}, 'n_clicks'),
    State('cause-effect-relations', 'children'),
    State('session-id', 'data')
)
def add_cause_effect(add_ce_clicks, remove_ce_clicks,relations, session_id):
    button_id = ctx.triggered_id if not None else 'No clicks yet'
    if button_id is None:
        return []
    
    if 'type' in button_id and 'index' in button_id and button_id['type'] == 'remove_ce':

        print('before rmoeve---------------------------------------------------------')

        print(relations)

        # for i,rm_click in enumerate(remove_ce_clicks):
        #     if rm_click is not None and rm_click>0:
        #         del relations[i]
        #         return relations
        relations = [rel for rel,rm_click in zip(relations,remove_ce_clicks) if rm_click is None]
        print('after rmoeve---------------------------------------------------------')
        print(relations)

        return relations#dbc.Container(relations)

    df = load_data(session_id)

    new_element = dbc.Container([
        dbc.Row([
            dbc.Col([dbc.Label("Cause:"), 
            dcc.Dropdown(id={'type': 'cause', 'index': add_ce_clicks}, value='',
                                                       options=[{'label': x, 'value': x} for x in df.columns])], width=2),

            dbc.Col([dbc.Label("Effect:"), 
            dcc.Dropdown(id={'type': 'effect', 'index': add_ce_clicks}, value='',
                                                        options=[{'label': x, 'value': x} for x in df.columns])], width=2),
            dbc.Col(dbc.Button(id={'type': 'remove_ce', 'index': add_ce_clicks},
                               className="bi bi-trash  rounded-circle m-4",outline=True, color="primary"), width=1)
        ],justify="right"
        ),html.Hr(),dbc.Tooltip(f"remove relation {button_id}",target={'type': 'remove_ce', 'index': add_ce_clicks})])
    
    relations.append(new_element)

    return relations



# @callback(
#     Output('test1', 'children'),
#     Input({'type': 'remove_ce', 'index': ALL}, 'n_clicks'),
#     prevent_initial_call=True
# )
# def test1(remove_ce_clicks):
#     for i,c in enumerate(remove_ce_clicks):
#         if c is not None and c>0:
#             return [dbc.Label(f"button_id: {i}")]
#     return [dbc.Label(f"no button clicked {len(remove_ce_clicks)} {remove_ce_clicks} {ctx.triggered_id}")]
    
#     button_id = ctx.triggered_id if not None else 'No clicks yet'
#     if button_id is None:
#         return []
#     return [dbc.Label(f"button_id: {button_id}")]


@cache.memoize()
def query_data(session_id):
    return pd.read_json(cache.get(session_id+'_data'))


@dash.callback(Output('model-res', 'children'),
               Input('build-model-button', 'n_clicks'),
               [State({'type': 'cause', 'index': ALL}, 'value'),
                State({'type': 'effect', 'index': ALL}, 'value'),
                State('session-id', 'data')],
               background=True,
               running=[
    (Output("build-model-button", "disabled"), True, False),
    (Output("cancel-build", "disabled"), False, True),
],
    cancel=[Input("cancel-build", "n_clicks")]
)
def process_data(n_clicks, causes,effects,session_id):
    if n_clicks is None :
        return None

    dd = json.dumps({'my_causes':causes,'my_effects':effects})
    return html.Pre(dd)


# @callback(
#     output=Output("paragraph_id", "children"),
#     inputs=Input("build-model-button", "n_clicks"),
#     state = [State('session-id', 'data')],
#     background=True,
#     running=[
#         (Output("build-model-button", "disabled"), True, False),
#          (Output("cancel_button_id", "disabled"), False, True),
#             ],
#             cancel=[Input("cancel_button_id", "n_clicks")]
#         )
# def update_clicks(n_clicks,session_id):
#     # if n_clicks is None:
#     #     return
#     time.sleep(2.0)
#     print('inside update_clicks')
#     return [f"Clicked {n_clicks} times m {session_id}"]
