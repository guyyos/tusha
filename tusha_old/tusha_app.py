import uuid
import dash_bootstrap_components as dbc
import dash
import dash_html_components as html
import dash_core_components as dcc
from dash.dependencies import Output, Input, State
from dash import no_update,ctx
import base64
import datetime
import io
import pandas as pd

# Connect to the layout and callbacks of each tab
from eda import eda_layout,gen_eda_children
from causal_model import causal_model_layout,get_causal_model_layout
from app import app,cache
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

    dbc.Input(
        type="text",
        id=dict(type='searchData'),
        placeholder="Enter Location",
        persistence=False,
        autocomplete="off",
        list='list-suggested-inputs1',
        className='rounded-pill'
    ),
    html.Datalist(id='list-suggested-inputs1',
                  children=[html.Option(value='empty')]),
    dbc.Button(
        "Enter commands", id="popover-bottom-target", color="info"
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
                                        activeLabelClassName="text-danger",children=eda_layout),
                                dbc.Tab(label="Causal Model", tab_id="tab-causal-model",
                                        labelClassName="text-success font-weight-bold", 
                                        activeLabelClassName="text-danger",children=causal_model_layout),
                            ],
                            id="tabs",
                            active_tab="tab-eda",
                        ),
                    ], className="mt-3"
                )


sidebar_ = dbc.Card(
    [
        dcc.Upload(
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
        dbc.Label("loaded file:", id='loaded_file'),
        commands_layout
    ],
    # width=2,
    # style=SIDEBAR_STYLE,
    # style={'position':'sticky','bottom':0},
    # body=True
)



sidebar = html.Div(
    [
        dmc.Affix(dbc.Button(chr(9776)+ ' open', id="open-offcanvas", n_clicks=0,className='rounded-pill'),
                  position={"bottom": 20, "left": 20}),
        dbc.Offcanvas(
            sidebar_,
            id="offcanvas",
            title="Title",
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
        dbc.Row(id='content', children=[]) ])


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


@app.callback(Output('eda_layout', 'children'),
              Input('upload-data', 'contents'),
              State('upload-data', 'filename'),
              State('upload-data', 'last_modified'),
              State('session-id', 'data'))
def update_output(contents, filename, last_modified,session_id):
        
    if contents:
        
        last_modified = datetime.datetime.fromtimestamp(last_modified)

        df = parse_file_contents(contents, filename)

        cache.set(session_id+'_data',df.to_json())
        cache.set(session_id+'_name',filename)

        eda_children = gen_eda_children(df,filename)
        return eda_children


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
