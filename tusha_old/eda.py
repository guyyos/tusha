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


eda_layout = html.Div(id='eda_layout')


def gen_eda_children(df,name):

    return html.Div(
        [
            dbc.Row(dbc.Col(html.H5(name), width=2), justify="center"),
            dbc.Container(id='eda_plots', children=[]),
            dbc.Row(dbc.Col(dbc.Button('add plot', id="eda-add-plot", n_clicks=0,className='rounded-pill'),width=2)),

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

@callback(
    Output('eda_plots', 'children'),
    Input('eda-add-plot', 'n_clicks'),
    State('eda_plots', 'children'),
    State('datatable-interactivity', "derived_virtual_data"),
)
def display_dropdowns(n_clicks, children, data):
    df = pd.DataFrame(data)

    new_element = dbc.Container([
                dbc.Row(
                    [
                        dbc.Col([dcc.Dropdown(id={'type': 'yaxis-data','index': n_clicks}, value='Body Mass (g)',
                                              options=[{'label': x, 'value': x} for x in df.columns])], width=2),
                        dbc.Col(html.Div(id={'type': 'output-div','index': n_clicks})),
                        dbc.Col([dbc.Label("size:"),
                                 dbc.Row(dcc.Dropdown(id={'type': 'size-data','index': n_clicks}, value='Body Mass (g)',
                                                      options=[{'label': x, 'value': x} for x in df.columns])),
                                 dbc.Label("color:"),
                                 dbc.Row(dcc.Dropdown(id={'type': 'color-data','index': n_clicks}, value='Island',
                                                      options=[{'label': x, 'value': x} for x in df.columns]))

                                 ], width=2),
                    ], align="center",),
                dbc.Row(dbc.Col(dcc.Dropdown(id={'type': 'xaxis-data','index': n_clicks}, value='Culmen Length (mm)',
                                             options=[{'label': x, 'value': x} for x in df.columns]), width=2), justify="center"),
                html.Hr(),  # horizontal line
            ])
    
    children.append(new_element)
    return children


@callback(Output({'type': 'output-div', 'index': MATCH}, 'children'),
          #   State('stored-data', 'data'),
          Input('datatable-interactivity', "derived_virtual_data"),
          Input({'type': 'xaxis-data', 'index': MATCH}, 'value'),
          Input({'type': 'yaxis-data', 'index': MATCH}, 'value'),
          Input({'type': 'size-data', 'index': MATCH}, 'value'),
          Input({'type': 'color-data', 'index': MATCH}, 'value'),
          )
def make_graphs(data, x_data, y_data, size_data, color_data):

    # import plotly.graph_objects as go
    # df = pd.DataFrame(data)
    # histY = go.Figure(data=[go.Histogram(y=df[y_data])])
    # histX = px.histogram(data, x=x_data)
    # data_ = data if rows is None else rows
    df = pd.DataFrame(data)
    mean, std = df[size_data].mean(), df[size_data].std()
    size_col = df[size_data].apply(lambda x: 5+3*(x-mean)/std)
    size_col = size_col.apply(lambda x: min(max(x, 1), 20))
    # scatt = px.scatter(data_,  x=x_data, y=y_data, size=size_data,color=color_data,
    #                    trendline="ols",marginal_x="histogram",marginal_y="histogram")
    scatt = px.scatter(data,  x=x_data, y=y_data, color=color_data, size=size_col,
                       trendline="ols", marginal_x="histogram", marginal_y="histogram")

    return dcc.Graph(figure=scatt)
