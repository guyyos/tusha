"""
This app creates a simple sidebar layout using inline style arguments and the
dbc.Nav component.

dcc.Location is used to track the current location, and a callback uses the
current location to render the appropriate page content. The active prop of
each NavLink is set automatically according to the current pathname. To use
this feature you must install dash-bootstrap-components >= 0.11.0.

For more details on building multi-page Dash applications, check out the Dash
documentation: https://dash.plot.ly/urls
"""
import dash
import dash_bootstrap_components as dbc
from dash import Input, Output, dcc, html, State
from eda import gen_eda_children
# from causal_model import causal_model_layout,get_causal_model_layout
from app import app,cache
import uuid
import datetime
import base64
import pandas as pd
import io

# app = dash.Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])

# the style arguments for the sidebar. We use position:fixed and a fixed width
SIDEBAR_STYLE = {
    "position": "fixed",
    "top": 0,
    "left": 0,
    "bottom": 0,
    "width": "16rem",
    "padding": "2rem 1rem",
    "background-color": "#f8f9fa",
}

# the styles for the main content position it to the right of the sidebar and
# add some padding.
CONTENT_STYLE = {
    "margin-left": "18rem",
    "margin-right": "2rem",
    "padding": "2rem 1rem",
}


sidebar = html.Div(
    [
        html.H2("Sidebar", className="display-4"),
        html.Hr(),
        html.P(
            "A simple sidebar layout with navigation links", className="lead"
        ),
        dbc.Button("Load data", id="load-data-button", n_clicks=0),
        html.Hr(),
        dbc.Label("loaded file:", id='loaded_file'),

        dbc.Container(
            children=[],id='dialog-open'
        ),
        html.Hr(),
        dbc.Container(children=[],id='navigation-explore'),
        dbc.Container(id="add-explore", children=[]),
        dcc.Store('num_explore_pages', data=1),
        dcc.Store(data=str(uuid.uuid4()), id='session-id')
    ],
    style=SIDEBAR_STYLE,
)

@app.callback(Output('add-explore','children'),
              Input('navigation-explore', 'children'))
def add_explore_btn(navs):
    if navs:
        return [html.Hr(),
                dbc.Button(id="add-explore-button", children='add explore',
                   outline=True, color="primary", n_clicks=0, disabled=False)]

@app.callback(Output('navigation-explore','children'),
              Input('loaded_file', 'children'))
def update_navigation(file_name):

    if file_name:

        return dbc.Nav(
                children=[
                    dbc.NavLink("Explore", id='explore_link_1',active="exact",
                                href="/explore-1"),
                ],
                id='navigation',
                vertical=True,
                pills=True,
            )


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

        return df
    except Exception as e:
        print(e)
        return html.Div([
            'There was an error processing this file.'
        ])
    

@app.callback(Output('loaded_file', 'children'),
              Output("close-load", "n_clicks"),
              Output('session-id','data'),
              Input('ok-load', 'n_clicks'),
              Input('upload-data', 'contents'),
              State('upload-data', 'filename'),
              State('upload-data', 'last_modified'),
              State('loaded_file', 'children'),
              State('session-id','data'))
def update_output(ok_clicks,contents, filename, last_modified,
                  cur_file,cur_session_id):

    print(f'ok_clicks {ok_clicks}')
        
    if ok_clicks:
        if filename:
            
            session_id = str(uuid.uuid4())
            print(f'session_id {session_id}')

            last_modified = datetime.datetime.fromtimestamp(last_modified)

            df = parse_file_contents(contents, filename)
            print(f'filename {filename}')
            print(f'df {df.head()}')

            cache.set(session_id+'_data',df.to_json())
            cache.set(session_id+'_name',filename)
            print(f' after set to cache')

            return filename,1,session_id
        return cur_file,1,cur_session_id
    return cur_file,0,cur_session_id
    

@app.callback(Output('selected_file', 'children'),
              Input('upload-data', 'contents'),
              State('upload-data', 'filename'),
              State('upload-data', 'last_modified'))
def update_output(contents, filename, last_modified):
        
    if contents:
        return filename



content = html.Div(id="page-content", style=CONTENT_STYLE)

app.layout = html.Div([dcc.Location(id="url"), sidebar, content])

@app.callback(
    Output("dialog-open", "children"),
    Input('load-data-button', "n_clicks")
)
def open_dialog_load_file(n_clicks):
    print(f'open_dialog_load_file {n_clicks}')
    if n_clicks:
        return dbc.Modal(
                [
                    # dbc.ModalHeader(dbc.ModalTitle("Header")),
                    dbc.ModalBody([dcc.Upload(
                            id='upload-data',
                            children=html.Div([
                                    'Drag and Drop or ',
                                    html.A('Select Files')
                                    ]),
                            style={
                                # 'width': '100%',
                                # 'height': '60px',
                                # 'lineHeight': '60px',
                                'borderWidth': '1px',
                                'borderStyle': 'dashed',
                                'borderRadius': '5px',
                                'textAlign': 'center',
                                # 'margin': '10px'
                            },
                            # DONT Allow multiple files to be uploaded
                            multiple=False,
                            className='rounded-pill'

                        ),
                        dbc.Label("file:", id='selected_file')]),
                
                    dbc.ModalFooter([
                        
                        dbc.Button(
                            "Ok", id="ok-load", className="btn btn-primary", n_clicks=0
                        ),
                        dbc.Button(
                            "Close", id="close-load", className="btn btn-secondary", n_clicks=0
                        )]
                    ),
                ],
                id="modal",
                is_open=True,
            )

@app.callback(
    Output("modal", "is_open"),
    Input("close-load", "n_clicks"),
    [State("modal", "is_open")],
)
def close_load_file_dialog(n_close, is_open):
    print(f'close_load_file_dialog {n_close}')
    if n_close and is_open:
        return False
    return is_open


def create_explore_page(pathname,session_id,exp_num):
        df = pd.read_json(cache.get(session_id+'_data'))
        filename = cache.get(session_id+'_name')

        eda_children = gen_eda_children(df,filename,exp_num)
        dash.register_page(pathname,  path=pathname, layout=eda_children)

        return eda_children

# html.Div([dbc.Label(f'page {exp_num}'), dcc.Dropdown(value='abc',options=['abc','ddd','eee'],id=f'drop1_{exp_num}',persistence =True),
#                          dbc.Container(children=[],id=f'temp_drop_{exp_num}')])



@app.callback(Output("page-content", "children"), 
              Input("url", "pathname"),
            State('session-id','data'))
def render_page_content(pathname,session_id):
    if pathname == "/":
        return html.Div([dbc.Label('home page')])
    elif pathname.startswith('/explore-'):
        exp_num = int(pathname.split('/explore-')[1])
        return create_explore_page(pathname,session_id,exp_num)
    
    # If the user tries to reach a different page, return a 404 message
    return html.Div(
        [
            html.H1("404: Not found", className="text-danger"),
            html.Hr(),
            html.P(f"The pathname {pathname} was not recognised..."),
        ],
        className="p-3 bg-light rounded-3",
    )


@app.callback(Output("temp_drop1", "children"), [Input("drop1", "value")])
def modify_temp_drop1(drop_val):
    if drop_val:
        return [dbc.Label(drop_val)]

@app.callback(Output("temp_drop2", "children"), [Input("drop2", "value")])
def modify_temp_drop2(drop_val):
    if drop_val:
        return [dbc.Label(drop_val)]


@app.callback(Output('navigation', 'children'),
              Output('num_explore_pages', 'data'),
              Input('add-explore-button', 'n_clicks'),
              State('navigation', 'children'),
              State('num_explore_pages', 'data'))
def add_view(n_clicks, navs, num_explore_pages):
    if n_clicks:
        num_explore_pages += 1
        navs.append(dbc.NavLink(
            f"Explore [{num_explore_pages}]", id=f'explore_{num_explore_pages}', href=f"/explore-{num_explore_pages}", active="exact"))
    return navs, num_explore_pages


if __name__ == "__main__":
    # app.run_server(port=5555)
    app.run_server(debug=True)

