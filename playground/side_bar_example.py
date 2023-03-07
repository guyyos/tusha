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

app = dash.Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])

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
        dcc.Store('num_pages', data=3)

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
    return dbc.Nav(
            children=[
                dbc.NavLink(f"Home {file_name}", id='pg_0', 
                            href="/", active='exact'),
                dbc.NavLink("Page 1", id='pg_1',active="exact",
                            href="/page-1"),
                dbc.NavLink("Page 2", id='pg_2',
                            href="/page-2", active="exact"),
            ],
            id='navigation',
            vertical=True,
            pills=True,
        ),


@app.callback(Output('loaded_file', 'children'),
              Output("close-load", "n_clicks"),
              Input('ok-load', 'n_clicks'),
              Input('upload-data', 'contents'),
              State('upload-data', 'filename'),
              State('upload-data', 'last_modified'),
              State('loaded_file', 'children'))
def update_output(ok_clicks,contents, filename, last_modified,cur_file):

    print(f'ok_clicks {ok_clicks}')
        
    if ok_clicks:
        if filename:
            return filename,1
        return cur_file,1
     




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



@app.callback(Output("page-content", "children"), [Input("url", "pathname")])
def render_page_content(pathname):
    if pathname == "/":
        return html.Div([dbc.Label('home page')])
    elif pathname == "/page-1":
        return html.Div([dbc.Label('page 1'), dcc.Dropdown(value='abc',options=['abc','ddd','eee'],id='drop1',persistence =True),
                         dbc.Container(children=[],id='temp_drop1')])
    elif pathname == "/page-2":
        return html.Div([dbc.Label('page 2'), dcc.Dropdown(value='abc',options=['abc','ddd','eee'],id='drop2',persistence =True),
                         dbc.Container(children=[],id='temp_drop2')])
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
              Output('num_pages', 'data'),
              Input('add-explore-button', 'n_clicks'),
              State('navigation', 'children'),
              State('num_pages', 'data'))
def add_view(n_clicks, navs, num_pages):
    if n_clicks:
        num_pages += 1
        navs.append(dbc.NavLink(
            f"Page {num_pages}", id=f'pg_{num_pages}', href="/page-{num_pages}}", active="exact"))
    return navs, num_pages


if __name__ == "__main__":
    app.run_server(port=8888)
