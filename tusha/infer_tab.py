import plotly.express as px
import dash_bootstrap_components as dbc
import dash
import dash_html_components as html
import dash_core_components as dcc
from dash.dependencies import Output, Input, State
from plot_creation import create_plots_with_reg_hdi_lines
from pandas import DataFrame
from file_handle import load_file,get_full_name
from multivar_model_creation import execute_model
import os


infer_layout = html.Div(id='infer_tab_layout',
                        children=[
                            html.Br(),

                            dbc.Row([

                                dbc.Col(dbc.Button(id="infer-model-button",
                                                   outline=True, color="primary",
                                                   n_clicks=0, className='bi bi-robot rounded-circle'), width=1),

                                dbc.Col(dbc.Button(id="download-model-button",
                                                   outline=True, color="primary",
                                                   n_clicks=0, className='bi bi-box-arrow-down rounded-circle'), width=1),

                                dbc.Col(dbc.Button(id="cancel-infer", outline=True, color="primary",
                                                   n_clicks=0, className='bi bi-x-lg rounded-circle'), width=1),

                                dbc.Col(dbc.Container(
                                    id='infer_model_spinner', children=[]), width=1)
                            ]),
                            dbc.Row(dbc.Toast(
                                header="No model to infer. Press build model first",
                                id="infer-model-err-msg",
                                icon="warning",
                                duration=4000,
                                is_open=False,
                            )),
                            dbc.Tooltip("Inference model",
                                        target="infer-model-button"),
                            dcc.Download(id="download-model"),

                            html.Hr(),
                            dbc.Container(id='model-infer-res', children=[])

                        ])


@dash.callback(Output('model-infer-res', 'children'),
               Output('infer-model-err-msg', "is_open"),
               Output('infer-model-err-msg', 'header'),
               Input('infer-model-button', 'n_clicks'),
               [State('session-id', 'data'),
                State('cause-effect-relations', 'data')],
               background=True,
               running=[
    (Output("infer-model-button", "disabled"), True, False),
    (Output("cancel-infer", "disabled"), False, True),
    (Output("infer_model_spinner", "children"), [dbc.Spinner(size="sm")], [])
],
    cancel=[Input("cancel-infer", "n_clicks")]
)
def infer_model(n_clicks, session_id, cause_effect_rels):
    if n_clicks <= 0:
        return [], False, ''

    if len(cause_effect_rels) <= 0:
        return [], True, 'Populate cause-effect table'

    btn = dash.callback_context.triggered[0]["prop_id"].split(".")[0]
    infer_figs = infer_model(session_id)
    return infer_figs, infer_figs == None, 'No model to infer yes. Press Build model.'


def infer_model(session_id):

    complete_model = load_file('complete_model', session_id)
    if complete_model is None:
        return None

    graph = load_file('graph', session_id)
    topo_order = load_file('topo_order', session_id)
    cat_num_map_per_target = load_file('cat_num_map_per_target', session_id)
    df = load_file('df', session_id)
    df1 = load_file('df1', session_id)

    figs = create_inference_figs(
        df, df1, complete_model, graph, topo_order, cat_num_map_per_target)

    return figs


def create_inference_figs(df, df1, complete_model, graph, topo_order, cat_num_map_per_target):
    all_figs = []

    model, res, summary_res, graph = execute_model(
        df1, complete_model, graph, topo_order, cat_num_map_per_target)

    figs = create_plots_with_reg_hdi_lines(df, summary_res, graph)

    for target, target_fig in figs.items():

        for predictor, pred_fig in target_fig.items():
            fig = pred_fig['fig']
            predictors = pred_fig['predictors']

            preds = ','.join(
                [f'<b>{p}</b>' if p == predictor else p for p in predictors])
            fig_name = f'[{preds}]<b>→{target}</b>'
            fig.update_layout(title=fig_name)
            all_figs.append(dcc.Graph(id=fig_name, figure=fig))

            if pred_fig['fig_mu']:
                fig = pred_fig['fig_mu']
                fig_name = f'mu: [{predictor}]<b>→{target}</b>'
                fig.update_layout(title=fig_name)
                all_figs.append(dcc.Graph(id=fig_name, figure=fig))

    return all_figs


@dash.callback(
    Output("download-model", "data"),
    Input("download-model-button", "n_clicks"),
    State('session-id', 'data'),
    prevent_initial_call=True,
)
def func(n_clicks,session_id):
    
    fname = get_full_name(session_id,'complete_model')

    if os.path.isfile(fname):
        return dcc.send_file(fname)
   
    # complete_model = load_file('complete_model', session_id)
    # encoded_data = base64.b64encode(complete_model).decode('utf-8')
    # return dcc.send_data(encoded_data, filename='model', mimetype='application/octet-stream')
