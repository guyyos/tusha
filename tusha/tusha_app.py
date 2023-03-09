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

BUTTON_STYLE = {'font-size': '16px', 'width': '120px', 'display': 'inline-block', 'margin-bottom': '10px', 'margin-right': '5px', 'height':'37px', 'verticalAlign': 'middle'}
UPLOAD_BUTTON_STYLE = dict(BUTTON_STYLE)
UPLOAD_BUTTON_STYLE['borderWidth'] = '1px'
UPLOAD_BUTTON_STYLE['borderStyle'] = 'dashed'
UPLOAD_BUTTON_STYLE['color'] = 'blue'
UPLOAD_BUTTON_STYLE['textAlign'] = 'center'
UPLOAD_BUTTON_STYLE['height'] = '30px'

commands_layout = html.Div([

    dbc.Col(dbc.Input(
        type="text",
        id=dict(type='searchData'),
        placeholder="Enter command",
        persistence=False,
        autocomplete="off",
        list='list-suggested-inputs1',
        className='rounded-pill'
    ),width={"size": 10, "offset": 1}),
    html.Datalist(id='list-suggested-inputs1',
                  children=[html.Option(value='empty')]),
    html.P(),
    dbc.Col(dbc.Button(
        " Execute", id="popover-bottom-target", className='bi bi-play rounded-pill',outline=True, color="primary",
        style = BUTTON_STYLE
    ),width={"size": 4, "offset": 1}),
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
                    dbc.Tab(label=" Data", tab_id="tab-eda", labelClassName="bi bi-clipboard-data",
                            activeLabelClassName="text-danger", children=eda_layout),
                    dbc.Tab(label=" Causal Model", tab_id="tab-causal-model",
                            labelClassName='bi bi-diagram-2',
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
        html.Br(),
        dbc.Col(dcc.Upload(
            id='upload-data',
            children=' Upload', #u"\U0001F4BE"+' Load',
            style = UPLOAD_BUTTON_STYLE,
            # style={
            #     # 'width': '100%',
            #     # 'height': '60px',
            #     # 'lineHeight': '60px',
            #     'borderWidth': '1px',
            #     'borderStyle': 'dashed',
            #     # 'borderRadius': '5px',
            #     'textAlign': 'center',
            #     'color':"blue",
            #     'outline':True
            #                     # 'margin': '10px'
            # },
            # DONT Allow multiple files to be uploaded
            multiple=False,
            className='bi bi-upload rounded-pill'

        ),width={"size": 3, "offset": '1'}),
        dbc.Col(dcc.Markdown(NO_FILE_LOADED_MSG, id='loaded_file'),width={"offset": 1}),
        html.Hr(),
        commands_layout,


        html.Hr(),
        dbc.Col(dbc.Button(' Data', id='change-to-eda-tab',
                   n_clicks=0, className='bi bi-clipboard-data rounded-pill',outline=True, color="primary",
                   style=BUTTON_STYLE),width={"size": 4, "offset": 1}),
        dbc.Container(children=[], id='quick-links-eda'),
        html.Hr(),
        dbc.Col(dbc.Button(' Causal', id='change-to-causal-tab',
                   n_clicks=0, className='bi bi-diagram-2 rounded-pill',outline=True, color="primary",
                   style = BUTTON_STYLE),width={"size": 4, "offset": 1}),
        dbc.Container(children=[], id='quick-links-causal'),

        # dbc.Nav(
        #     [
        #         dbc.NavLink("Mid", href="#mid_causal", external_link=True),
        #         dbc.NavLink("End", href="#end_causal", external_link=True),
        #     ],
        #     vertical=True,
        #     pills=True,
        # ),


    ],
    # width=2,
    # style=SIDEBAR_STYLE,
    # style={'position':'sticky','bottom':0},
    # body=True
)


@app.callback(Output('quick-links-eda', 'children'),
              Input('eda_plots', 'children')
              )
def update_quick_links_eda(eda_plots):

    if eda_plots:

        return [
            dbc.Nav(
                [dbc.NavLink(id = {'type': 'eda-plot-link', 'index': i},children=f"plot- {i}", 
                             href=f"#{eda_plot['props']['id']}",external_link=True)
                 for i, eda_plot in enumerate(eda_plots)]+\
                    [dbc.NavLink(f"Data Table", href=f"#datatable-interactivity", external_link=True)],
                vertical=True,
                pills=True,
            )]


@app.callback(Output('quick-links-causal', 'children'),
              Input('model-res', 'children')
            )
def update_quick_link_univar_causal(model_res_plots):

    causal_plot_links = [dbc.NavLink(id = f"link_{causal_plot['props']['id']}",children=f"{causal_plot['props']['id']}", 
                            href=f"#{causal_plot['props']['id']}",external_link=True)
                for i, causal_plot in enumerate(model_res_plots)] if model_res_plots else []
    
    return [
        dbc.Nav(causal_plot_links,
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
            title="Commands & links",
            is_open=False
        ),
    ]
)


def get_content():
    return dbc.Container([
        dbc.Row([dbc.Col([dbc.Row(html.H2(" Tusha",
                        style={"textAlign": "left"},className='fa-solid fa-cat')),dbc.Row(dbc.Label('Data Causal Assistant'))])]),
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


@app.callback(Output('eda_layout', 'children'),
              Output('loaded_file', 'children'),
              Output('change-to-eda-tab', 'n_clicks'),
              Input('upload-data', 'contents'),
              State('upload-data', 'filename'),
              State('upload-data', 'last_modified'),
              State('session-id', 'data'))
def update_output(contents, filename, last_modified, session_id):

    if contents:

        last_modified = datetime.datetime.fromtimestamp(last_modified)

        df = parse_file_contents(contents, filename)

        cache.set(session_id+'_data', df.to_json())
        cache.set(session_id+'_name', filename)

        eda_children = gen_eda_children(df, filename)
        return eda_children,f'**{filename}**',1
    return None, NO_FILE_LOADED_MSG,0


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
    # app.run_server(debug=True)
    app.run(host='0.0.0.0', port=8050,debug=True)