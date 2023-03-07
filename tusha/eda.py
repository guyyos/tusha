import plotly.express as px
import dash_bootstrap_components as dbc
import dash
import dash_html_components as html
import dash_core_components as dcc
from dash.dependencies import Output, Input, State
import pandas as pd
import dash_table
from dash import dcc, html, callback, Output, Input
import seaborn as sns
from matplotlib import pyplot as plt
from dash import Dash, dcc, html, Input, Output, State, MATCH, ALL
import json


eda_layout = html.Div(id='eda_layout')


def gen_eda_children(df, name):

    return html.Div(
        [
            dbc.Row(dbc.Col(html.H5(name), width=2), justify="center"),
            dbc.Container(id='eda_plots', children=[]),
            html.P("", id="mid_eda"),
            dbc.Row(dbc.Col(dbc.Button(' Add plot', id="eda-add-plot",
                    n_clicks=0, className='bi bi-graph-up-arrow rounded-pill',outline=True, color="primary"), width=2)),

            # dcc.Store(id='stored-data', data=df.fillna(df.mean()).to_dict('records')),

            # dash_table.DataTable(
            #     data=df.to_dict('records'),
            #     columns=[{'name': i, 'id': i} for i in df.columns],
            #     page_size=15
            # ),
            html.Br(),  # horizontal line

            dash_table.DataTable(
                id='datatable-interactivity',
                columns=[
                        {"name": i, "id": i, "deletable": True, "selectable": True} for i in df.columns
                ],
                data=df.fillna(df.mean()).to_dict('records'),
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
                style_table={
                    'overflowY': 'scroll'
                }

            ), 

        ]
    )


@callback(Output('eda_plots', 'children'),
          Input('eda-add-plot', 'n_clicks'),
          Input({'type': 'remove-plot', 'index': ALL}, 'n_clicks'),
          State('eda_plots', 'children'),
          State('datatable-interactivity', "derived_virtual_data")
          )
def modify_eda_plots(n_clicks, remove_click, children, data):
    print(f'modify_eda_plots {dash.callback_context.triggered}')
    btn = dash.callback_context.triggered[0]["prop_id"].split(".")[0]
    print(f'modify_eda_plots btn {btn}')

    if btn == "eda-add-plot":
        return create_new_plot(children, data)
    elif btn.startswith('{') and btn.endswith('}'):
        btn_id = json.loads(btn)
        remove_id = None
        if btn_id['type'] == 'remove-plot':
            for i,child in enumerate(children):
                child_id = child['props']['id']
                if child_id == btn_id["index"]:
                    remove_id = i
                    break
            if remove_id is not None:
                del children[i]
        return children

    return children

import uuid

def create_new_plot(children, data):
    df = pd.DataFrame(data)

    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    numeric_columns = df.select_dtypes(include=numerics).columns


    idx = str(uuid.uuid4())

    new_element = dbc.Container([
        dbc.Row(
            [
                dbc.Col([dbc.Label("y:"),dcc.Dropdown(id={'type': 'yaxis-data', 'index': idx}, value=None,
                                      options=[{'label': x, 'value': x} for x in df.columns])], width=2),
                dbc.Col(
                    html.Div(id={'type': 'output-div', 'index': idx})),
                dbc.Col([dbc.Label("size:"),
                         dbc.Row(dcc.Dropdown(id={'type': 'size-data', 'index': idx}, value=None,disabled=True,
                                              options=[{'label': x, 'value': x} for x in numeric_columns])),
                         dbc.Label("color:"),
                         dbc.Row(dcc.Dropdown(id={'type': 'color-data', 'index': idx}, value=None,
                                              options=[{'label': x, 'value': x} for x in df.columns]))
                         ], width=2),
            ], align="center",),
        dbc.Row([dbc.Col([dbc.Label("x:"),dcc.Dropdown(id={'type': 'xaxis-data', 'index': idx}, value=None,
                                      options=[{'label': x, 'value': x} for x in df.columns])], width=2)
                ], justify="center"),
        dbc.Row([dbc.Col(dbc.Button(id={'type': 'remove-plot', 'index': idx},
                                    n_clicks=0, className='bi bi-trash3 rounded-circle', outline=True, color="primary"),width=1)],align='right'),
        html.Hr(),  # horizontal line
    ],id=idx)

    if children is None:
        children = []

    children.append(new_element)

    return children

@callback(Output({'type': 'eda-plot-link', 'index': ALL}, 'children'),
          Input({'type': 'xaxis-data', 'index': ALL}, 'value'),
          Input({'type': 'yaxis-data', 'index': ALL}, 'value'),
          Input({'type': 'size-data', 'index': ALL}, 'value'),
          Input({'type': 'color-data', 'index': ALL}, 'value'),
          State({'type': 'eda-plot-link', 'index': ALL}, 'children')
          )
def modify_quick_link(x_datas, y_datas, size_data, color_data,plot_links):

    new_plot_links = [f"{x_data}-{y_data}" if x_data else 'plot' for x_data,y_data in zip(x_datas, y_datas)]

    if len(plot_links)!= len(new_plot_links):
        print(f'modify_quick_link plot_links {plot_links}')
        print(f'modify_quick_link new_plot_links {new_plot_links}')
        return plot_links
    return new_plot_links


@callback(Output({'type': 'output-div', 'index': MATCH}, 'children'),
          Output({'type': 'yaxis-data', 'index': MATCH}, 'disabled'),
          Output({'type': 'size-data', 'index': MATCH}, 'disabled'),
          Output({'type': 'color-data', 'index': MATCH}, 'disabled'),
          Input('datatable-interactivity', "derived_virtual_data"),
          Input({'type': 'xaxis-data', 'index': MATCH}, 'value'),
          Input({'type': 'yaxis-data', 'index': MATCH}, 'value'),
          Input({'type': 'size-data', 'index': MATCH}, 'value'),
          Input({'type': 'color-data', 'index': MATCH}, 'value'),
          )
def make_graphs(data, x_data, y_data, size_data, color_data):

    print(f'make_graphs x_data = {x_data}')
    print(f'make_graphs y_data = {y_data}')
    print(f'make_graphs size_data = {size_data}')
    print(f'make_graphs color_data = {color_data}')
    if x_data is None:
        return None,True,True,True
    
    df = pd.DataFrame(data)

    if df[x_data].dtype.name == 'object':
        if y_data and df[y_data].dtype.name == 'object':
            graph = px.bar(df, x=x_data, color=y_data)
            return dcc.Graph(figure=graph),False,True,True

        elif y_data:#numeric
            graph = px.box(df, x=x_data, y=y_data, color=color_data,notched=True)
            return dcc.Graph(figure=graph),False,True,False
        else:
            graph = px.bar(df, x=x_data)
            return dcc.Graph(figure=graph),False,True,False

    #x numerical
    if y_data: 
        size_col = None
        if size_data:

            mean, std = df[size_data].mean(), df[size_data].std()
            size_col = df[size_data].apply(lambda x: 5+3*(x-mean)/std)
            size_col = size_col.apply(lambda x: min(max(x, 1), 20))
        
        print(f'make_graphs before px.scatter')

        graph = px.scatter(data,  x=x_data, y=y_data, color=color_data, size=size_col,
                    trendline="ols", marginal_x="histogram", marginal_y="histogram")
    else:
        graph = px.histogram(data, x=x_data)

    return dcc.Graph(figure=graph),False,False,False
