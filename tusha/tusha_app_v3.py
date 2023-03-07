import uuid
import dash_bootstrap_components as dbc
import dash
import dash_html_components as html
import dash_core_components as dcc
from dash.dependencies import Output, Input, State
from dash import no_update, ctx
import base64
import datetime
import io
import pandas as pd

# Connect to the layout and callbacks of each tab
from eda import eda_layout, gen_eda_children
from causal_model import causal_model_layout, get_causal_model_layout
from app import app, cache
import dash_mantine_components as dmc


# the style arguments for the sidebar. We use position:fixed and a fixed width
SIDEBAR_STYLE = {
    "position": "fixed",
    "top": 0,
    "left": 0,
    "bottom": 0,
    "width": "14rem",
    "padding": "2rem 1rem",
    # "background-color": "#f8f9fa",
}

# the styles for the main content position it to the right of the sidebar and
# add some padding.
CONTENT_STYLE = {
    "margin-left": "5%",
    # "margin-right": "1px",
    'padding-right': '30px',
    # "padding": "2rem 1rem",
}

commands_layout = html.Div([

    dbc.Col(dbc.Input(
        type="text",
        id=dict(type='searchData'),
        placeholder="Enter command",
        persistence=False,
        autocomplete="off",
        list='list-suggested-inputs1',
        className='rounded-pill'
    ),width=10),
    html.Datalist(id='list-suggested-inputs1',
                  children=[html.Option(value='empty')]),
    html.P(),
    dbc.Button(
        "Execute", id="popover-bottom-target", className='rounded-pill',outline=True, color="primary"
    ),
    dbc.Popover(
        [
            # dbc.PopoverHeader("commands111:"),
            # dbc.PopoverBody('make your commands here'),
            dbc.Input(type="text", id=dict(type='searchData', id='dest-loc'),
                      placeholder="Eg. Load data.csv", autocomplete="off", list='list-suggested-inputs'),
            html.Datalist(id='list-suggested-inputs',
                          children=[html.Option(value='empty')]),
            html.Hr(),
            dbc.Label("Previous Commands:", className="mr-2"),

        ],
        id="popover",
        target="popover-bottom-target",  # needs to be the same as dbc.Button id
        placement="bottom",
        is_open=False,
    ),
]
)


@app.callback(
    Output('list-suggested-inputs', 'children'),
    Input({"id": 'dest-loc', "type": "searchData"}, "value"),
    prevent_initial_call=True
)
def suggest_locs(value):
    if len(value) < 4:
        raise dash.exceptions.PreventUpdate
    # google_api_key ='YOUR KEY'
    # url = f'https://maps.googleapis.com/maps/api/place/autocomplete/json?input={value}&types=locality&region=us&key={google_api_key}'
    # r = requests.get(url)

    predictions = ['guyy', 'florida', 'florida yes']
    print(predictions)

    return [html.Option(value=l) for l in predictions]


@app.callback(
    Output("popover", "is_open"),
    [Input("popover-bottom-target", "n_clicks")],
    [State("popover", "is_open")],
)
def toggle_popover(n, is_open):
    if n:
        return not is_open
    return is_open


# our app's Tabs *********************************************************
def get_app_tabs():

    return html.Div(
        [
            dcc.Store(data=str(uuid.uuid4()), id='session-id'),

            dbc.Tabs(
                [
                    dbc.Tab(label="Explore", tab_id="tab-eda", labelClassName="text-success font-weight-bold",
                            activeLabelClassName="text-danger", children=eda_layout),
                    dbc.Tab(label="Causal Model", tab_id="tab-causal-model",
                            labelClassName="text-success font-weight-bold",
                            activeLabelClassName="text-danger", children=causal_model_layout),
                ],
                id="tabs",
                active_tab="tab-eda",
            ),
        ], className="mt-3"
    )


@app.callback(Output('tabs', 'active_tab'),
              [Input('change-to-eda-tab', 'n_clicks'),
              Input('change-to-causal-tab', 'n_clicks')])
def on_change_tab_click(click1, click2):

    btn = dash.callback_context.triggered[0]["prop_id"].split(".")[0]

    if btn == "change-to-eda-tab":
        return "tab-eda"
    elif btn == "change-to-causal-tab":
        return "tab-causal-model"

NO_FILE_LOADED_MSG = 'No file loaded'

sidebar_ = dbc.Card(
    [
        dbc.Col(dbc.Button("Load", id="load-data-button", n_clicks=0,className='rounded-pill',outline=True, color="primary"),width=3),
        html.Hr(),
        dbc.Label("", id='loaded_file'),

        dbc.Container(
            children=[],id='dialog-open'
        ),

        html.Hr(),
        commands_layout,


        html.Hr(),
        dbc.Col(dbc.Button('Explore', id='change-to-eda-tab',
                   n_clicks=0, className='rounded-pill',outline=True, color="primary"),width=3),
        dbc.Container(children=[], id='quick-links-eda'),
        html.Hr(),
        dbc.Col(dbc.Button('Causal', id='change-to-causal-tab',
                   n_clicks=0, className='rounded-pill',outline=True, color="primary"),width=3),
        dbc.Nav(
            [
                dbc.NavLink("Mid", href="#mid_causal", external_link=True),
                dbc.NavLink("End", href="#end_causal", external_link=True),
            ],
            vertical=True,
            pills=True,
        ),


    ],
    # width=2,
    # style=SIDEBAR_STYLE,
    # style={'position':'sticky','bottom':0},
    # body=True
)


@app.callback(Output('quick-links-eda', 'children'),
              Input('eda_plots', 'children')
              )
def update_causl_model(eda_plots):

    if eda_plots:

        return [
            dbc.Nav(
                [dbc.NavLink(f"plot- {i}", href=f"#{eda_plot['props']['id']}", external_link=True)
                 for i, eda_plot in enumerate(eda_plots)]+\
                    [dbc.NavLink(f"data", href=f"#datatable-interactivity", external_link=True)],
                vertical=True,
                pills=True,
            )]


sidebar = html.Div(
    [
        dmc.Affix(dbc.Button(chr(9776) + ' open', id="open-offcanvas", n_clicks=0, className='rounded-pill', outline=True, color="primary"),
                  position={"bottom": 20, "left": 20}),
        dbc.Offcanvas(
            sidebar_,
            id="offcanvas",
            title="Tusha",
            is_open=False,
        ),
    ]
)


def get_content():
    return dbc.Container([
        dbc.Row(html.H1("Tusha Analytics",
                        style={"textAlign": "center"})),
        html.Hr(),
        dbc.Row(get_app_tabs()),
        dbc.Row(id='content', children=[])])


def serve_layout():
    return html.Div([
        dcc.Location(id="url"),
        get_content(),
        sidebar])


app.layout = serve_layout


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



@app.callback(Output('loaded_file', 'children'),
              Output("close-load", "n_clicks"),
              Output('session-id','data'),
              Input('ok-load', 'n_clicks'),
              Input('upload-data', 'contents'),
              State('upload-data', 'filename'),
              State('upload-data', 'last_modified'),
              State('loaded_file', 'children'),
              State('session-id','data'),
              prevent_initial_call=True
              )
def update_output(ok_clicks,contents, filename, last_modified,
                  cur_file,cur_session_id):

    print(f'update_output ok_clicks ={ok_clicks}')
        
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
        return None,None,None
    return None,None,None



@app.callback(Output('selected_file', 'children'),
              Input('upload-data', 'contents'),
              State('upload-data', 'filename'),
              State('upload-data', 'last_modified'))
def update_selected_file(contents, filename, last_modified):
        
    if contents:
        return filename



@app.callback(Output('eda_layout', 'children'),
              Output('change-to-eda-tab', 'n_clicks'),
              Input('loaded_file', 'children'),
              State('session-id', 'data'),
              State('eda_layout', 'children'))
def update_output(filename, session_id, cur_eda_children):

    if filename:
        print(f'update_output filename:{filename}')
        
        df = pd.read_json(cache.get(session_id+'_data'))

        eda_children = gen_eda_children(df, filename)
        return eda_children,1
    return cur_eda_children,0




@app.callback(Output('causal_model_layout', 'children'),
              Input('eda_layout', 'children'),
              State('session-id', 'data'))
def update_causl_model(eda_updated, session_id):

    if eda_updated:

        df = pd.read_json(cache.get(session_id+'_data'))

        eda_children = get_causal_model_layout(df)
        return eda_children


@app.callback(
    Output("offcanvas", "is_open"),
    Input("open-offcanvas", "n_clicks"),
    [State("offcanvas", "is_open")],
)
def toggle_offcanvas(n1, is_open):
    if n1:
        return not is_open
    return is_open


if __name__ == '__main__':
    app.run_server(debug=True)
