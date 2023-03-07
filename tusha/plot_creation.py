import plotly.graph_objects as go
import plotly.express as px


def create_plots_with_reg_hdi_lines(df,target,predictors,all_res):
    figs = {}

    for predictor in predictors:
        target_lower = all_res[predictor][0][:,0]
        target_higher = all_res[predictor][0][:,1]

        mu_lower = all_res[predictor][1][:,0]
        mu_higher = all_res[predictor][1][:,1]

        mu_mean = all_res[predictor][2]

        predictor_vals = all_res[predictor][3]

        other_predictor = list(set(predictors).difference(set([predictor])))[0]
        figs[predictor] = create_scatter_with_reg_hdi_lines(df,target,predictor,other_predictor,target_lower,target_higher,mu_lower,mu_higher,mu_mean,predictor_vals)

    return figs


def create_scatter_with_reg_hdi_lines(df,target,predictor,other_predictor,target_lower,target_higher,mu_lower,mu_higher,mu_mean,predictor_vals):

    # df[f'scaled_[{other_predictor}]'] = df[other_predictor].apply(lambda x: 5+3*(x-df[other_predictor].mean())/df[other_predictor].std())
    # df[f'scaled_[{other_predictor}]'] = df[f'scaled_[{other_predictor}]']-df[f'scaled_[{other_predictor}]'].min()

    fig = px.scatter(df,  x=predictor, y=target, color=other_predictor, size=None,
                    trendline="ols")

    # fig = go.Figure()
    # fig.add_trace(go.Scatter(x=df[predictor], y=df[target],
    #     name="items",fill=None,mode='markers'
    #     ))


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

