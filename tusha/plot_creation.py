import plotly.graph_objects as go
import plotly.express as px
from pandas import DataFrame
from base_def import FeatureType


def create_plots_with_reg_hdi_lines(df,summary_res,graph):
    figs = {}

    for target,target_res in summary_res.items():
        figs[target] = {}

        for predictor,prediction_summary in target_res.items():
            target_type = graph.nodes[target].info.featureType
            predictor_type = graph.nodes[predictor].info.featureType
            fig,fig_mu = create_plot_with_reg_hdi_lines(df,target,predictor,prediction_summary,target_type,predictor_type)

            figs[target][predictor] = {}
            figs[target][predictor]['fig'] = fig
            figs[target][predictor]['fig_mu'] = fig_mu
            figs[target][predictor]['predictors'] = prediction_summary.predictors
    
    return figs



def create_plot_with_reg_hdi_lines(df,target,predictor,prediction_summary,target_type,predictor_type):
    fig_mu = None

    if predictor_type.is_categorical(): #if len(prediction_summary.cat_codes)>0:
        data = []
        for cat_val,vals,mu_vals in zip(prediction_summary.cat_codes,
                                        prediction_summary.target_pred.values,prediction_summary.mu_pred.values):
            for v,mv in zip(vals,mu_vals):
                data.append({predictor:cat_val,target:v,f'{target}_mu':mv})

        df_vals = DataFrame(data)

        fig = px.box(df_vals, x=predictor, y=target)
        fig_mu = px.box(df_vals, x=predictor, y=f'{target}_mu')

    else:

        target_lower = prediction_summary.target_hdi[:,0]
        target_higher = prediction_summary.target_hdi[:,1]
        mu_lower = prediction_summary.mu_hdi[:,0]
        mu_higher = prediction_summary.mu_hdi[:,1]
        mu_mean = prediction_summary.mu_mean
        predictor_vals = prediction_summary.predictor_vals

        other_predictors = list(set(prediction_summary.predictors).difference(set([predictor])))
        other_predictor = other_predictors[0] if len(other_predictors)>0 else None

        target_is_numerical = target_type == FeatureType.NUMERICAL
        fig = create_scatter_with_reg_hdi_lines(df,target,predictor,other_predictor,target_lower,
                        target_higher,mu_lower,mu_higher,mu_mean,predictor_vals,target_is_numerical)

    return fig,fig_mu


# def create_box_plot():
#     graph = px.box(df, x=x_data, y=y_data, color=color_data,notched=True)


def create_scatter_with_reg_hdi_lines(df,target,predictor,other_predictor,target_lower,target_higher,mu_lower,mu_higher,mu_mean,predictor_vals,target_is_numerical):

    if target_is_numerical:
        fig = px.scatter(df,  x=predictor, y=target, color=other_predictor, size=None,
                    trendline="ols")
    else:
        fig = px.scatter(df,  x=predictor, y=target, color=other_predictor, size=None)
    
    fig.add_trace(go.Scatter(x=predictor_vals, y=target_lower,
        showlegend=False,
        fill=None,
        mode='lines',
        line=dict(width=0.5, color='rgb(184, 247, 212)'),
        ))
    fig.add_trace(go.Scatter(x=predictor_vals, y=target_higher,
        showlegend=False,
        name="hdi",
        fill='tonexty', # fill area between trace0 and trace1
        mode='lines', line=dict(width=0.5, color='rgb(184, 247, 212)'),))

    if target_is_numerical:

        fig.add_trace(go.Scatter(x=predictor_vals, y=mu_higher,
            showlegend=False,
            fill=None,
            mode='lines',
            line=dict(width=0.5, color='rgb(111, 100, 255)'),
            ))

        fig.add_trace(go.Scatter(x=predictor_vals, y=mu_lower,
            showlegend=False,
            name="trend_range",
            fill='tonexty', # fill area between trace0 and trace1
            mode='lines', line=dict(width=0.5, color='rgb(111, 100, 255)')))


        fig.add_trace(go.Scatter(x=predictor_vals, y=mu_mean,
            showlegend=False,
            name="trend",
            mode='lines', line_color='black'))
    
    return fig

def create_scatter_with_reg_line(df,smooth_df,target,predictor):

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df[predictor], y=df[target],
        name="items",
        fill=None,mode='markers'
        ))


    fig.add_trace(go.Scatter(x=smooth_df[predictor], y=smooth_df[f'{target}_hdi_lower'],
        showlegend=False,
        fill=None,
        mode='lines',
        line=dict(width=0.5, color='rgb(184, 247, 212)'),
        ))
    fig.add_trace(go.Scatter(
    x=smooth_df[predictor], y=smooth_df[f'{target}_hdi_upper'],
        name="hdi",
        fill='tonexty', # fill area between trace0 and trace1
        mode='lines', line=dict(width=0.5, color='rgb(184, 247, 212)'),))

    fig.add_trace(go.Scatter(x=smooth_df[predictor], y=smooth_df[f'mu_{target}_hdi_lower'],
        showlegend=False,
        fill=None,
        mode='lines',
        line=dict(width=0.5, color='rgb(111, 100, 255)'),
        ))
    fig.add_trace(go.Scatter(
    x=smooth_df[predictor], y=smooth_df[f'mu_{target}_hdi_upper'],
        name="trend_range",
        fill='tonexty', # fill area between trace0 and trace1
        mode='lines', line=dict(width=0.5, color='rgb(111, 100, 255)')))


    fig.add_trace(go.Scatter(
    x=smooth_df[predictor], y=smooth_df[f'mu_{target}_mid'],
        name="trend",
        mode='lines', line_color='black'))

    return fig

