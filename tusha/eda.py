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
import uuid
from load_data_tab import load_data


eda_layout = html.Div(id='eda_layout')


@callback(Output('eda_layout', 'children'),
          Input('cur_data_file','data'),
          State('session-id', 'data'))
def update_layout(filename, session_id):
    print(f'update_layout filename = {filename}')
    if filename:

        df,time_col = load_data(session_id)

        data_children = gen_eda_children(df, filename, time_col)

        return data_children
    return None


def gen_eda_children(df, name, time_col):

    return html.Div(
        [
            dbc.Row(dbc.Col(html.H5(name), width=2), justify="center"),
            dbc.Container(id='eda_plots', children=[]),
            html.P("", id="mid_eda"),
            dbc.Row([dbc.Col(dbc.Button(' Add plot', id="eda-add-plot",
                    n_clicks=0, className='bi bi-graph-up-arrow rounded-pill', outline=True, color="primary"), width=2),
                    dbc.Col(dbc.Button(' Add time plot', id="eda-add-time-plot",
                                       n_clicks=0, className='bi bi-graph-up-arrow rounded-pill', outline=True, color="primary", disabled=not time_col), width=2)])
        ]
    )


@callback(Output('eda_plots', 'children'),
          Input('eda-add-plot', 'n_clicks'),
          Input('eda-add-time-plot', 'n_clicks'),
          Input({'type': 'remove-plot', 'index': ALL}, 'n_clicks'),
          State('eda_plots', 'children'),
          State('session-id', 'data'))
def modify_eda_plots(n_clicks, n_clicks1, remove_click, children,session_id):
    data,time_col = load_data(session_id)
    print(f'modify_eda_plots {dash.callback_context.triggered}')
    btn = dash.callback_context.triggered[0]["prop_id"].split(".")[0]
    print(f'modify_eda_plots btn {btn}')

    if btn == "eda-add-plot":
        return create_new_plot(children, data)
    if btn == "eda-add-time-plot":
        return create_new_time_plot(children, data, time_col)
    elif btn.startswith('{') and btn.endswith('}'):
        btn_id = json.loads(btn)
        remove_id = None
        if btn_id['type'] == 'remove-plot':
            for i, child in enumerate(children):
                child_id = child['props']['id']
                if child_id == btn_id["index"]:
                    remove_id = i
                    break
            if remove_id is not None:
                del children[i]
        return children

    return children


def create_new_time_plot(children, data, time_col):
    df = pd.DataFrame(data)

    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    numeric_columns = list(df.select_dtypes(include=numerics).columns)
    categorical_columns = list(df.select_dtypes(exclude=numerics).columns)

    if time_col in numeric_columns:
        numeric_columns.remove(time_col)
    if time_col in categorical_columns:
        categorical_columns.remove(time_col)

    idx = str(uuid.uuid4())
    
    def create_options(vals):
        return [{"value":val,"label":html.Span(val, style={"font-size": 15, "padding-left": 10})} 
                                                                       for val in vals]

    symbol_col = [dcc.Markdown("**Symbol:**" if categorical_columns else ''), dbc.Row(dcc.RadioItems(
        id={'type': 'symbol-data-time', 'index': idx}, options=create_options(categorical_columns), value=None,
        labelStyle={"display": "flex", "align-items": "center"}))]


    new_element = dbc.Container([
        dbc.Row(
            [
                dbc.Col([dcc.Markdown("**Metrics:**"), dcc.Checklist(id={'type': 'yaxis-data-time', 'index': idx},
                                                              options=create_options(numeric_columns), value=[],
                                                              labelStyle={"display": "flex", "align-items": "center"})], width=2),
                dbc.Col(
                    html.Div(id={'type': 'output-div-time', 'index': idx})),
                dbc.Col(symbol_col, width=2),
            ], align="center"),
        html.Br(),
        dbc.Row([dbc.Col(dbc.Button(id={'type': 'remove-plot', 'index': idx},
                                    n_clicks=0, className='bi bi-trash3 rounded-circle', outline=True, color="primary"), width=1)], align='right'),
        html.Hr(),  # horizontal line
        dcc.Store(id={'type': 'plot_info', 'index': idx}, data='plot')
    ], id={'type': 'graph_container', 'index': idx})

    new_element = dbc.Container(new_element, id=idx)

    if children is None:
        children = []

    children.append(new_element)

    return children


def create_new_plot(children, data):
    df = pd.DataFrame(data)

    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    numeric_columns = df.select_dtypes(include=numerics).columns

    idx = str(uuid.uuid4())

    new_element = dbc.Container([
        dbc.Row(
            [
                dbc.Col([dcc.Markdown("**Y:**"), dcc.Dropdown(id={'type': 'yaxis-data', 'index': idx}, value=None,
                                                       options=[{'label': x, 'value': x} for x in df.columns])], width=2),
                dbc.Col(
                    html.Div(id={'type': 'output-div', 'index': idx})),
                dbc.Col([dcc.Markdown("**Size:**"),
                         dbc.Row(dcc.Dropdown(id={'type': 'size-data', 'index': idx}, value=None, disabled=True,
                                              options=[{'label': x, 'value': x} for x in numeric_columns])),
                         html.Br(),
                         dcc.Markdown("**Color:**"),
                         dbc.Row(dcc.Dropdown(id={'type': 'color-data', 'index': idx}, value=None,
                                              options=[{'label': x, 'value': x} for x in df.columns]))
                         ], width=2),
            ], align="center"),
        dbc.Row([dbc.Col([dcc.Markdown("**X:**"), dcc.Dropdown(id={'type': 'xaxis-data', 'index': idx}, value=None,
                                                        options=[{'label': x, 'value': x} for x in df.columns])], width=2)
                 ], justify="center"),
        dbc.Row([dbc.Col(dbc.Button(id={'type': 'remove-plot', 'index': idx},
                                    n_clicks=0, className='bi bi-trash3 rounded-circle', outline=True, color="primary"), width=1)], align='right'),
        html.Hr(),  # horizontal line
        dcc.Store(id={'type': 'plot_info', 'index': idx}, data='plot')
    ], id={'type': 'graph_container', 'index': idx})

    new_element = dbc.Container(new_element, id=idx)

    if children is None:
        children = []

    children.append(new_element)

    return children


def find_element_in_props_tree(props_tree, id_type):
    if type(props_tree) is dict and 'id' in props_tree and \
            type(props_tree['id']) is dict and 'type' in props_tree['id'] and props_tree['id']['type'] == id_type:
        return props_tree
    if type(props_tree) is dict:
        for el in props_tree.values():
            res = find_element_in_props_tree(el, id_type)
            if res:
                return res
    if type(props_tree) is list:
        for el in props_tree:
            res = find_element_in_props_tree(el, id_type)
            if res:
                return res
    return None


@callback(Output({'type': 'plot_info', 'index': ALL}, 'data'),
          Input({'type': 'plot_info', 'index': ALL}, 'data'),
          Input({'type': 'graph_container', 'index': ALL}, 'children'),
          Input({'type': 'output-div', 'index': ALL}, 'children'),
          Input({'type': 'output-div-time', 'index': ALL}, 'children')
          )
def modify_plot_info(plot_infos, graph_containers, graphs, time_graphs):
    # print(f'modify_plot_info graph_containers = {graph_containers}')
    # print(f'modify_plot_info plot_infos = {plot_infos}')

    plot_infos = []
    for i, graph_container in enumerate(graph_containers):
        xelement = find_element_in_props_tree(graph_container, 'xaxis-data')

        if xelement and 'value' in xelement and xelement['value']:
            plot_info = f'Plot: {xelement["value"]}'
            yelement = find_element_in_props_tree(
                graph_container, 'yaxis-data')
            if yelement and 'value' in yelement and yelement['value']:
                plot_info = f'{plot_info} vs {yelement["value"]}'
            plot_infos.append(plot_info)
            continue

        yelement = find_element_in_props_tree(
            graph_container, 'yaxis-data-time')
        if yelement and 'value' in yelement and yelement['value']:
            plot_info = f"Time Plot: {','.join(yelement['value'])}"
            plot_infos.append(plot_info)
            continue

        plot_infos.append(f'Plot: {i}')

    return plot_infos


@callback(Output({'type': 'output-div', 'index': MATCH}, 'children'),
          Output({'type': 'yaxis-data', 'index': MATCH}, 'disabled'),
          Output({'type': 'size-data', 'index': MATCH}, 'disabled'),
          Output({'type': 'color-data', 'index': MATCH}, 'disabled'),
          Input({'type': 'xaxis-data', 'index': MATCH}, 'value'),
          Input({'type': 'yaxis-data', 'index': MATCH}, 'value'),
          Input({'type': 'size-data', 'index': MATCH}, 'value'),
          Input({'type': 'color-data', 'index': MATCH}, 'value'),
          State('session-id', 'data')
          )
def make_graphs( x_data, y_data, size_data, color_data,session_id):

    print(f'make_graphs x_data = {x_data}')
    print(f'make_graphs y_data = {y_data}')
    print(f'make_graphs size_data = {size_data}')
    print(f'make_graphs color_data = {color_data}')

    if x_data is None:
        return None, True, True, True
    
    df,time_col = load_data(session_id)

    if df[x_data].dtype.name == 'object':
        if y_data and df[y_data].dtype.name == 'object':
            graph = px.bar(df, x=x_data, color=y_data)
            return dcc.Graph(figure=graph), False, True, True

        elif y_data:  # numeric
            graph = px.box(df, x=x_data, y=y_data,
                           color=color_data, notched=True)
            return dcc.Graph(figure=graph), False, True, False
        else:
            graph = px.bar(df, x=x_data)
            return dcc.Graph(figure=graph), False, True, False

    # x numerical
    if y_data:
        size_col = None
        if size_data:

            mean, std = df[size_data].mean(), df[size_data].std()
            size_col = df[size_data].apply(lambda x: 5+3*(x-mean)/std)
            size_col = size_col.apply(lambda x: min(max(x, 1), 20))

        print(f'make_graphs before px.scatter')

        graph = px.scatter(df,  x=x_data, y=y_data, color=color_data, size=size_col,
                           trendline="ols", marginal_x="histogram", marginal_y="histogram")
    else:
        graph = px.histogram(df, x=x_data)

    return dcc.Graph(figure=graph), False, False, False


@callback(Output({'type': 'output-div-time', 'index': MATCH}, 'children'),
          Input({'type': 'yaxis-data-time', 'index': MATCH}, 'value'),
          Input({'type': 'symbol-data-time', 'index': MATCH}, 'value'),
          State('session-id', 'data')
          )
def make_time_graphs(y_data, symbol_data, session_id):

    print(f'make_time_graphs y_data = {y_data}')
    print(f'make_time_graphs symbol_data = {symbol_data}')

    if not y_data:
        return []

    df,time_col = load_data(session_id)
    symbols = [sym for symbol in ['circle','square','hexagram','star', 'diamond', 'hourglass', 'bowtie'] for sym in [symbol,f'{symbol}-open']]

    fig = px.line(df, x=time_col, y=y_data, symbol=symbol_data,symbol_sequence = symbols)

    return dcc.Graph(figure=fig)


