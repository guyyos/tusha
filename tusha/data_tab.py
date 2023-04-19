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



NO_FILE_LOADED_MSG = 'No file loaded'


data_layout = html.Div(id='data_tab_layout',
        children=[])


@callback(Output('data_tab_layout', 'children'),
          Output('cur_loaded_file','children'),
          Input('prev_file_selector', 'options'),
          Input('prev_file_selector', 'value'),
          State('session-id', 'data'))
def update_data(prev_files, filename, session_id):
    print(f'update_data prev_files = {prev_files}')
    print(f'update_data filename = {filename}')
    if filename:

        df,time_col = load_data(session_id)
        print(f'data_tab: update_data time_col = {time_col}')

        data_children = gen_data_children(df, filename, time_col)

        fname = f'**{filename}**'
        return data_children,fname
    return None,''



def gen_data_children(df, name, time_col):

    return html.Div(
        [
            dbc.Row(dbc.Col(html.H5(name), width=2), justify="center"),
            dbc.Container(id='data_plots', children=[]),
            html.P("", id="mid_data"),
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