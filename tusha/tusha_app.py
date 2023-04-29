import uuid
import dash_bootstrap_components as dbc
import dash
import dash_html_components as html
import dash_core_components as dcc
from dash.dependencies import Output, Input, State
from dash import no_update, ctx
import datetime
import io
import pandas as pd
from dash import Dash, dcc, html, Input, Output, State, MATCH, ALL


# Connect to the layout and callbacks of each tab
from eda import eda_layout, gen_eda_children
from data_tab import data_layout
from overview_tab import overview_layout
from load_data_tab import load_data_layout
from infer_tab import infer_layout
from causal_model import causal_model_layout
from app import app, cache
import dash_mantine_components as dmc
from identify_features import find_datetime_col


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
                    dbc.Tab(label=" Load", tab_id="tab-load-data", labelClassName="bi bi-minecart-loaded",
                            activeLabelClassName="text-danger", children=load_data_layout),
                    dbc.Tab(label=" Data", tab_id="tab-data", labelClassName="bi bi-list-columns",
                            activeLabelClassName="text-danger", children=data_layout),
                    dbc.Tab(label=" Overview", tab_id="tab-overview", labelClassName="bi bi-clipboard-data",
                            activeLabelClassName="text-danger", children=overview_layout),
                    dbc.Tab(label=" Explore", tab_id="tab-eda", labelClassName="bi bi-binoculars",
                            activeLabelClassName="text-danger", children=eda_layout),
                    dbc.Tab(label=" Causal Model", tab_id="tab-causal-model",
                            labelClassName='bi bi-diagram-2',
                            activeLabelClassName="text-danger", children=causal_model_layout),
                    dbc.Tab(label=" Inference", tab_id="tab-inference-model",
                            labelClassName='bi bi-robot',
                            activeLabelClassName="text-danger", children=infer_layout)
                ],
                id="tabs",
                active_tab="tab-load-data",
            ),
        ], className="mt-3"
    )


@app.callback(Output('tabs', 'active_tab'),
              [Input('tabs', 'active_tab'),Input('change-to-load-data-tab', 'n_clicks'),
               Input('change-to-data-tab', 'n_clicks'),
               Input('change-to-overview-tab', 'n_clicks'),
               Input('change-to-eda-tab', 'n_clicks'),
              Input('change-to-causal-tab', 'n_clicks'),
              Input('data_tab_layout','children'),
              Input('infer_tab_layout','children')])
def on_change_tab_click(cur_active_tab,click0,click1, click2, click3,click4,data_children,infer_children):

    btn = dash.callback_context.triggered[0]["prop_id"].split(".")[0]
    print(f'on_change_tab_click cur_active_tab = {cur_active_tab}')
    print(f'on_change_tab_click btn = {btn}')

    if data_children:
        print(f'on_change_tab_click data_children = {len(data_children)}')
    if infer_children:
        print(f'on_change_tab_click infer_children = {infer_children}')
    
    if btn == "change-to-load-data-tab":
        return "tab-load-data"
    if btn == "change-to-data-tab":
        return "tab-data"
    if btn == "change-to-overview-tab":
        return "tab-overview"
    if btn == "change-to-eda-tab":
        return "tab-eda"
    if btn == "change-to-causal-tab":
        return "tab-causal-model"
    
    if btn == 'data_tab_layout' and data_children:
        return 'tab-data'

    if btn == 'infer_tab_layout' and infer_children:
        return 'tab-inference-model'
    
    
    return cur_active_tab


sidebar_ = dbc.Card(
    [
        html.Br(),
        # dbc.Col(dcc.Upload(
        #     id='upload-data',
        #     children=' Upload', #u"\U0001F4BE"+' Load',
        #     style = UPLOAD_BUTTON_STYLE,
        #     # style={
        #     #     # 'width': '100%',
        #     #     # 'height': '60px',
        #     #     # 'lineHeight': '60px',
        #     #     'borderWidth': '1px',
        #     #     'borderStyle': 'dashed',
        #     #     # 'borderRadius': '5px',
        #     #     'textAlign': 'center',
        #     #     'color':"blue",
        #     #     'outline':True
        #     #                     # 'margin': '10px'
        #     # },
        #     # DONT Allow multiple files to be uploaded
        #     multiple=False,
        #     className='bi bi-upload rounded-pill'

        # ),width={"size": 3, "offset": '1'}),
        # dbc.Col(dcc.Markdown('', id='loaded_file'),width={"offset": 1}),
        # html.Hr(),
        # commands_layout,
        # html.Hr(),
        dbc.Col(dbc.Button(' Load', id='change-to-load-data-tab',
                   n_clicks=0, className='bi bi-minecart-loaded rounded-pill',outline=True, color="primary",
                   style=BUTTON_STYLE),width={"size": 4, "offset": 1}),
        html.Hr(),
        dbc.Col(dbc.Button(' Data', id='change-to-data-tab',
                   n_clicks=0, className='bi bi-list-columns rounded-pill',outline=True, color="primary",
                   style=BUTTON_STYLE),width={"size": 4, "offset": 1}),
        dbc.Container(children=[], id='quick-links-data'),
        html.Hr(),
        dbc.Col(dbc.Button(' Overview', id='change-to-overview-tab',
                   n_clicks=0, className='bi bi-clipboard-data rounded-pill',outline=True, color="primary",
                   style=BUTTON_STYLE),width={"size": 4, "offset": 1}),
        dbc.Container(children=[], id='quick-links-overview'),
        html.Hr(),
        dbc.Col(dbc.Button(' Explore', id='change-to-eda-tab',
                   n_clicks=0, className='bi bi-binoculars rounded-pill',outline=True, color="primary",
                   style=BUTTON_STYLE),width={"size": 4, "offset": 1}),
        dbc.Container(children=[], id='quick-links-eda'),
        html.Hr(),
        dbc.Col(dbc.Button(' Causal', id='change-to-causal-tab',
                   n_clicks=0, className='bi bi-diagram-2 rounded-pill',outline=True, color="primary",
                   style = BUTTON_STYLE),width={"size": 4, "offset": 1}),
        dbc.Container(children=[], id='quick-links-causal')
    ]
)

@app.callback(Output('quick-links-data', 'children'),
              Input('overview_plots', 'children')
              )
def update_quick_links_data(overview_plots):

    return [dbc.Nav([dbc.NavLink(f"Data Table", href=f"#datatable-interactivity", external_link=True)],vertical=True,
                pills=True)]


@app.callback(Output('quick-links-overview', 'children'),
              Input({'type': 'overview_plot_info', 'index': ALL}, 'key')
              )
def update_quick_links_overview(overview_plots):
    print(f'update_quick_links_overview overview_plots {overview_plots}')

    return [
            dbc.Nav(
                [dbc.NavLink(id = {'type': 'overview-plot-link', 'index': i},children=overview_plot, 
                             href=f"#{overview_plot}",external_link=True)
                 for i, overview_plot in enumerate(overview_plots)],
                vertical=True,
                pills=True,
            )]



@app.callback(Output('quick-links-eda', 'children'),
              Input('eda_plots', 'children')
              )
def update_quick_links_eda(eda_plots):
    print(f'update_quick_links_eda eda_plots')

    if eda_plots:

        return [
            dbc.Nav(
                [dbc.NavLink(id = {'type': 'eda-plot-link', 'index': i},children=f"plot- {i}", 
                             href=f"#{eda_plot['props']['id']}",external_link=True)
                 for i, eda_plot in enumerate(eda_plots)],
                vertical=True,
                pills=True,
            )]
    return []


@app.callback(Output({'type': 'eda-plot-link', 'index': ALL}, 'children'),
          Input({'type': 'plot_info', 'index': ALL}, 'data')
            )
def modify_quick_link(plot_infos):
    print(f'modify_quick_link: plot_infos = {plot_infos}')

    new_plot_links = plot_infos

    # if len(plot_links)!= len(new_plot_links):
    #     print(f'modify_quick_link plot_links {plot_links}')
    #     print(f'modify_quick_link new_plot_links {new_plot_links}')
    #     return plot_links
    return new_plot_links



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
        dmc.Affix(dbc.Button(chr(9776) + ' Navigate', id="open-offcanvas", n_clicks=0, className='rounded-pill', outline=True, color="primary"),
                  position={"bottom": 20, "left": 20}),
        dbc.Offcanvas(
            sidebar_,
            id="offcanvas",
            title="Navigate",
            is_open=False
        ),
    ]
)


def get_content():
    return dbc.Container([
        dbc.Row(html.H1('')),dbc.Row(html.H1('')),
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