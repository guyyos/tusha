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
import datetime
from identify_features import find_datetime_col
from app import cache
import base64
import io
from load_data_tab import load_data


overview_layout = html.Div(id='overview_tab_layout',
        children=[])


@callback(Output('overview_tab_layout', 'children'),
          Input('prev_file_selector', 'options'),
          Input('prev_file_selector', 'value'),
          State('session-id', 'data'))
def update_overview(prev_files, filename, session_id):
    print(f'update_overview prev_files = {prev_files}')
    print(f'update_overview filename = {filename}')
    if filename:

        df,time_col = load_data(session_id)
        print(f'overview_tab: update_overview time_col = {time_col}')

        overview_children = gen_overview_children(df, filename, time_col)

        return overview_children
    return None



def gen_overview_children(df, name, time_col):

    return html.Div(
        [
            dbc.Row(dbc.Col(html.H5(name), width=2), justify="center"),
            dbc.Container(id='overview_plots', children=get_overview_plots(df, time_col))          

        ])


def get_overview_plots(df, time_col):

    if time_col:
        return get_overview_time_plots(df, time_col)

    table_header = [
    html.Thead(html.Tr([html.Th("Feature"), html.Th("Type"), html.Th("Distribution")]))
    ]

    rows = []

    for col in df.columns:

        if df[col].dtype.name == 'object':
            col_type = 'Categorical'
            graph = px.bar(df, x=col,width=600, height=300) #,width=600, height=300 width=800, height=400
        else:
            col_type = 'Numerical'
            graph = px.histogram(df, x=col, width=600, height=300)
        
        rows.append(html.Tr([html.Td(col), html.Td(col_type), html.Td(dcc.Graph(figure=graph))]))

    table_body = [html.Tbody(rows)]

    table = dbc.Table(table_header + table_body, bordered=True)

    return table

def get_overview_time_plots(df, time_col):
    table_header = [
    html.Thead(html.Tr([html.Th("Feature"), html.Th("Type"), html.Th("Distribution"),html.Th("Over time")]))
    ]

    rows = []

    for col in df.columns:

        if df[col].dtype.name == 'object':
            col_type = 'Categorical'
            graph = px.bar(df, x=col,width=600, height=300) 
            time_graph = px.scatter(df,  x=time_col, y=col,width=600, height=300)
        else:
            col_type = 'Numerical'
            graph = px.histogram(df, x=col, width=600, height=300 )
            time_graph = px.scatter(df,  x=time_col, y=col,width=600, height=300)

        if col == time_col:
            col_type = 'Time'

        rows.append(html.Tr([html.Td(col), html.Td(col_type), html.Td(dcc.Graph(figure=graph)),  html.Td(dcc.Graph(figure=time_graph))]))

    table_body = [html.Tbody(rows)]

    table = dbc.Table(table_header + table_body, bordered=True)

    return table
