import plotly.graph_objects as go
import plotly.express as px
from pandas import DataFrame


def create_plots_with_reg_hdi_lines(df,target,predictors,pred_res):
    figs = {}

    for predictor in predictors:
        ps = pred_res[predictor]

        if len(ps.cat_codes)>0:
            #categorical

            data = []
            for cat_val,vals,mu_vals in zip(ps.cat_codes,ps.target_pred.values,ps.mu_pred.values):
                for v,mv in zip(vals,mu_vals):
                    data.append({predictor:cat_val,target:v,f'{target}_mu':mv})

            df_vals = DataFrame(data)

            figs[predictor] = px.box(df_vals, x=predictor, y=target)
            figs[f'{predictor}_mu'] = px.box(df_vals, x=predictor, y=f'{target}_mu')

        else:

            target_lower = ps.target_hdi[:,0]
            target_higher = ps.target_hdi[:,1]

            mu_lower = ps.mu_hdi[:,0]
            mu_higher = ps.mu_hdi[:,1]

            mu_mean = ps.mu_mean

            predictor_vals = ps.predictor_vals

            other_predictors = list(set(predictors).difference(set([predictor])))
            other_predictor = other_predictors[0] if len(other_predictors)>0 else None
            figs[predictor] = create_scatter_with_reg_hdi_lines(df,target,predictor,other_predictor,target_lower,target_higher,mu_lower,mu_higher,mu_mean,predictor_vals)

    return figs


# def create_box_plot():
#     graph = px.box(df, x=x_data, y=y_data, color=color_data,notched=True)


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

