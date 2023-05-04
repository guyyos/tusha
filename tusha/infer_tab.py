import plotly.express as px
import dash_bootstrap_components as dbc
import dash
import dash_html_components as html
import dash_core_components as dcc
from dash.dependencies import Output, Input, State
from plot_creation import create_plots_with_reg_hdi_lines
from pandas import DataFrame
from file_handle import load_file, get_full_name,save_file
from multivar_model_creation import calc_counterfactual_analysis,sample_model
import os
from app import cache


infer_model_component = html.Div(
    [
        html.Div(
            dcc.Markdown("Run inference simulation on our server (**might be slow**):")), html.Br(),
        dbc.Row(children=[dbc.Col(dbc.Button(id="infer-model-button",
                                             outline=True, color="primary",
                                             n_clicks=0, className='bi bi-robot rounded-circle'), width=5),

                          dbc.Col(dbc.Button(id="cancel-infer", outline=True, color="primary",
                                             n_clicks=0, className='bi bi-x-lg rounded-circle'), width=5),
                          dbc.Col(dbc.Container(
                              id='infer_model_spinner', children=[]), width=1)]),
        html.Br(),
        dcc.Interval(id="progress-interval", n_intervals=0, interval=500,disabled=True),
        dbc.Progress(id="progress")
    ],
    className="p-3 m-2 border",
)


download_step1 = html.Li(dbc.Row([dbc.Col(dcc.Markdown('Download model file (_model.bin_) and execution notebook (_infer.ipynb_):'), width=10), 
                                  dbc.Col(dbc.Button(id="download-model-button",outline=True, 
                                                     color="primary",n_clicks=0, className='bi bi-box-arrow-down rounded-circle'), width=1),
                                                     dcc.Download(id="download-model"),dcc.Download(id="infer-notebook")]))

download_step2 = html.Li(['Execute the notebook on your machine.'])

download_step3 = html.Li(dbc.Row([dbc.Col(dcc.Markdown('Upload simulation results file (_results.bin_): '), width=10),
                     dbc.Col(dcc.Upload(
                         id='upload-sim-results',
                         children=[dbc.Button(id="upload-sim-button",
                                              outline=True, color="primary",
                                              n_clicks=0, className='bi bi-cloud-upload rounded-circle')]), width=1)]))

download_step4 = dbc.Row([dbc.Col(dcc.Markdown(id='loaded_sim_file',children=''), width=10)])

download_explained = html.Ol(children=[download_step1,html.Br(),download_step2,html.Br(),download_step3,download_step4])

download_model_component = html.Div(
    [
        html.Div(
            dcc.Markdown("**OR**: Download the model file/execute on your machine/upload simulation results to view:")), html.Br(),

        dbc.Button(
            "More info",
            id="expand-download-model-button",
            className="mb-3",
            color="primary",
            n_clicks=0,
        ),
        dbc.Collapse(
            dbc.Card(dbc.CardBody(download_explained)),
            id="download_explained",
            is_open=False,
        ),
    ], className="p-3 m-2 border"
)


@dash.callback(Output('loaded_sim_file','children'),
               Input('upload-sim-results', 'contents'),
            State('upload-sim-results', 'filename'),
            State('upload-sim-results', 'last_modified'),
            State('session-id', 'data'))
def upload_sim_file(contents, filename, last_modified, session_id):
    print(f'upload_sim_file filename = {filename}')
    print(f'upload_sim_file last_modified = {last_modified}')
    if contents:
        import base64
        import io
        import pickle

        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)        

        dd = pickle.loads(decoded)

        if type(dd)==dict and 'new_model' in dd and 'idata' in dd:
                
            new_model = dd["new_model"]
            idata = dd["idata"]

            print(f'upload_sim_file type(new_model) = {type(new_model)}')
            print(f'upload_sim_file type(idata) = {type(idata)}')

            save_file('model_after_sampling', session_id, new_model)
            save_file('idata', session_id, idata)

            return f'**Loaded** _{filename}_'
        
        return 'invalid file'

def load_sampled_model_objs(session_id):
    
    model_after_sampling = load_file('model_after_sampling', session_id)
    idata = load_file('idata', session_id)

    return model_after_sampling,idata


@dash.callback(
    Output("download_explained", "is_open"),
    Output('expand-download-model-button', 'children'),
    [Input("expand-download-model-button", "n_clicks")],
    [State("download_explained", "is_open")],
)
def toggle_collapse(n, is_open):
    if not n:
        return is_open, 'More info'
    button_txt = 'More info' if is_open else 'Hide'
    return not is_open, button_txt


infer_layout = html.Div(id='infer_tab_layout',
                        children=[
                            html.Br(),

                            dbc.Row([
                                dbc.Col(infer_model_component, width=5),
                                dbc.Col(download_model_component, width=5)
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
                            html.Hr(),
                            dbc.Container(id='model-infer-res', children=[])

                        ])


@dash.callback(
    [Output("progress", "value"), Output("progress", "label")],
    [Input("progress-interval", "n_intervals")],
    State('session-id', 'data'),
)
def update_progress(n,session_id):

    progress = cache.get(session_id+'_sample_model_progress')
    if progress is None:
        return progress,''

    # only add text after 5% progress to ensure text isn't squashed too much
    return progress, f"{progress} %" if progress >= 5 else ""


@dash.callback(Output('model-infer-res', 'children'),
               Output('infer-model-err-msg', "is_open"),
               Output('infer-model-err-msg', 'header'),
               Input('infer-model-button', 'n_clicks'),
               Input('loaded_sim_file', 'children'),
               Input('prev_file_selector', 'value'),
               [State('session-id', 'data'),
                State('cause-effect-relations', 'data')],
               background=True,
               running=[
    (Output("infer-model-button", "disabled"), True, False),
    (Output("cancel-infer", "disabled"), False, True),
    (Output("infer_model_spinner", "children"), [dbc.Spinner(size="sm")], []),
    (Output("progress", "style"),{"visibility": "visible"},{"visibility": "hidden"}),
    (Output("progress-interval", "disabled"), False, True),
],
    cancel=[Input("cancel-infer", "n_clicks")]
)
def infer_model(n_clicks, loaded_sim_file,file_updated,session_id, cause_effect_rels):

    btn = dash.callback_context.triggered[0]["prop_id"].split(".")[0]
    print(f'infer_model btn {btn}')

    if btn == "prev_file_selector":
        return None, False, None

    if len(cause_effect_rels) <= 0:
        return [], True, 'Populate cause-effect table'

    res = load_model_objs(session_id)
    if res is None:
        return None, True, 'Error: Build a model before!'
    
    model,graph,topo_order,cat_num_map_per_target,df,df1 = res


    if btn == 'infer-model-button':        
        
        model_after_sampling,idata = sample_model(model,session_id)
    else:
        model_after_sampling,idata = load_sampled_model_objs(session_id)
        if model_after_sampling is None:
            return None, True, 'Error: No model simulations file to load!'

    infer_figs = create_inference_figs(
        model_after_sampling, idata,df, df1, graph, topo_order, cat_num_map_per_target)

    return infer_figs, infer_figs == None, 'No model to infer yes. Press Build model.'


def load_model_objs(session_id):

    model = load_file('model', session_id)
    if model is None:
        return None

    graph = load_file('graph', session_id)
    topo_order = load_file('topo_order', session_id)
    cat_num_map_per_target = load_file('cat_num_map_per_target', session_id)
    df = load_file('df', session_id)
    df1 = load_file('df1', session_id)

    return model,graph,topo_order,cat_num_map_per_target,df,df1


def create_inference_figs(model_after_sampling, idata,df, df1, graph, topo_order, cat_num_map_per_target):
    all_figs = []

    model, res, summary_res, graph = calc_counterfactual_analysis(df1,model_after_sampling, idata,graph, topo_order, cat_num_map_per_target)

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
    Output("infer-notebook","data"),
    Input("download-model-button", "n_clicks"),
    State('session-id', 'data'),
    prevent_initial_call=True,
)
def download_model(n_clicks, session_id):

    fname = get_full_name(session_id, 'model')

    if os.path.isfile(fname):
        return dcc.send_file(fname),dcc.send_file('tusha/infer.ipynb')
    
    return None,None
    
