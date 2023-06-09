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
from app import cache, UPLOAD_DIRECTORY,EXAMPLES_DIRECTORY
import base64
import io
import os


NO_FILE_LOADED_MSG = 'No file loaded'


prev_file_selector = html.Div(
    [
        html.Div("Or select from previous files:"),
        dcc.Dropdown(
            id='prev_file_selector',
            options=[],
            value=''
        )

    ],
    className="p-3 m-2 border",
)

def list_dir_files(dirname):

    files = []
    for filename in os.listdir(dirname):
        path = os.path.join(dirname, filename)
        if os.path.isfile(path):
            files.append(filename)
    return files


def list_examples_files():
    """List the files in the upload directory."""

    examples_files = list_dir_files(EXAMPLES_DIRECTORY)
    print(f'examples_files = {examples_files}')
    return examples_files


examples_selector = html.Div(
    [
        html.Div("Try an example:"),
        dcc.Dropdown(
            id='example_selector',
            options=list_examples_files(),
            value=''
        )

    ],
    className="p-3 m-2 border",
)


cur_loaded_file = html.Div(
    [
        html.Div("Loaded data file:"),
        dcc.Markdown(
            id='cur_loaded_file',
            children=''
        )

    ],
    className="p-3 m-2 border",
)


load_data_layout = html.Div(id='load_data_layout',
                            children=[
                                dbc.Row(dcc.Upload(
                                    id='upload-data',
                                    children=html.Div([
                                        'Drag and Drop or ',
                                        html.A('Select Files')
                                    ]),
                                    style={
                                        'width': '100%',
                                        'height': '60px',
                                        'lineHeight': '60px',
                                        'borderWidth': '1px',
                                        'borderStyle': 'dashed',
                                        'borderRadius': '5px',
                                        'textAlign': 'center',
                                        'margin': '10px'
                                    },
                                    # Allow multiple files to be uploaded
                                    multiple=False
                                )), 
                                dbc.Row([dbc.Col(prev_file_selector,width=3),dbc.Col(examples_selector,width=3)]),
                                dbc.Row(dbc.Col(cur_loaded_file,width=3)),
                                dcc.Store(data={}, id='cur_data_file')])


@callback(Output('prev_file_selector', 'options'),
          Output('prev_file_selector', 'value'),
          Output('example_selector','value'),
          Output('cur_data_file','data'),
          Input('upload-data', 'contents'),
          Input('example_selector','value'),
          Input('prev_file_selector', 'value'),
          State('upload-data', 'filename'),
          State('upload-data', 'last_modified'),
          State('session-id', 'data'))
def upload_file(contents,example, prev_filename, filename, last_modified, session_id):
    btn = dash.callback_context.triggered[0]["prop_id"].split(".")[0]
    print(f'upload_file btn {btn}')
    print(f'update_output filename = {filename}')
    print(f'update_output last_modified = {last_modified}')
    print(f'update_output example = {example}')

    if btn == 'example_selector':
        filename = example
        df = load_example(filename)
        save_df(filename, df, session_id)
    elif btn == 'prev_file_selector':
        filename = prev_filename
    elif contents is None:
        return [],'','',''
    else:
        save_file(filename, contents, session_id)

    cache.set(session_id+'_name', filename)
    df = load_session_file(filename,session_id)
    
    time_col = find_datetime_col(df)
    
    cache.set(session_id+'_data', df.to_json())
    cache.set(session_id+'_time_col', time_col)

    files = list_uploaded_files(session_id)

    return files,filename,'',filename


@callback(Output('chosen_file', 'children'),
          Input('uploaded-files', 'value')
          )
def upload_file_option(uploaded_file):

    print(f'upload_file_option = {uploaded_file}')

    return uploaded_file


def parse_file_contents(contents, filename):
    content_type, content_string = contents.split(',')

    decoded = base64.b64decode(content_string)
    try:
        if 'csv' in filename:
            # Assume that the user uploaded a CSV file
            df = pd.read_csv(
                io.StringIO(decoded.decode('utf-8')))
        elif 'xls' in filename:
            # Assume that the user uploaded an excel file
            df = pd.read_excel(io.BytesIO(decoded))

        preprocess_df(df)

        return df
    except Exception as e:
        print(e)
        return html.Div([
            'There was an error processing this file.'
        ])


def preprocess_df(df):

    from pandas.errors import ParserError
    for c in df.columns[df.dtypes == 'object']:  # don't cnvt num
        try:
            df[c] = pd.to_datetime(df[c])
        except (ParserError, ValueError):  # Can't cnvrt some
            pass  # ...so leave whole column as-is unconverted
    return df


def get_upload_dir(session_id):
    return UPLOAD_DIRECTORY+'/'+session_id


def load_data(session_id):
    df = pd.read_json(cache.get(session_id+'_data'))
    time_col = cache.get(session_id+'_time_col')
    preprocess_df(df)
    return df,time_col


def load_session_file(filename,session_id):
    dirname = get_upload_dir(session_id)
    fullname = os.path.join(dirname, filename)
    return load_file(fullname)

def load_example(example):
    fullname = os.path.join(EXAMPLES_DIRECTORY, example)
    return load_file(fullname)


def load_file(fullname):
    try:
        if '.csv' in fullname:
            # Assume that the user uploaded a CSV file
            df = pd.read_csv(fullname)
        elif '.xls' in fullname:
            # Assume that the user uploaded an excel file
            df = pd.read_excel(fullname)

        preprocess_df(df)

        return df
    except Exception as e:
        print(e)
        return html.Div([
            'There was an error processing this file.'
        ])



def save_file(name, content, session_id):
    dirname = get_upload_dir(session_id)
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    """Decode and store a file uploaded with Plotly Dash."""
    data = content.encode("utf8").split(b";base64,")[1]
    with open(os.path.join(dirname, name), "wb") as fp:
        fp.write(base64.decodebytes(data))


def save_df(name, df, session_id):
    dirname = get_upload_dir(session_id)
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    
    full_name = os.path.join(dirname, name)
    df.to_csv(full_name,index=False)



def list_uploaded_files(session_id):
    """List the files in the upload directory."""
    dirname = get_upload_dir(session_id)

    return list_dir_files(dirname)

