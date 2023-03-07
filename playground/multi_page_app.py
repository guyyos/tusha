from dash import Dash, html, dcc, callback
import dash
import dash_bootstrap_components as dbc
import uuid


app = Dash(__name__, use_pages=True, pages_folder="")

dash.register_page("home",  path='/', layout=html.Div('Home Page'))
dash.register_page("analytics", layout=html.Div('Analytics'))

app.layout = html.Div([
    html.Div(
        [
            html.Div(
                dcc.Link(
                    f"{page['name']} - {page['path']}", href=page["relative_path"]
                )
            )
            for page in dash.page_registry.values()
        ]
    ),
    dash.page_container,
])


app.layout = html.Div(
    [
        dcc.Store(data=str(uuid.uuid4()), id='session-id'),

        dbc.Tabs(
            [
                dbc.Tab(label="Explore", tab_id="tab-eda", labelClassName="text-success font-weight-bold",
                        activeLabelClassName="text-danger", children=[]),
                dbc.Tab(label="Causal Model", tab_id="tab-causal-model",
                        labelClassName="text-success font-weight-bold",
                        activeLabelClassName="text-danger", children=[]),
                dbc.Tab(label='mode_tabs', tab_id='more-tabs', children=[html.Div(
                    [
                        html.Div(dcc.Link(f"{page['name']} - {page['path']}", href=page["relative_path"])
                                 )
                        for page in dash.page_registry.values()
                    ]
                ),
                    dash.page_container])],
            id="tabs",
            active_tab="tab-eda",
        ),
    ], className="mt-3"
)


if __name__ == '__main__':
    app.run_server(debug=True)
