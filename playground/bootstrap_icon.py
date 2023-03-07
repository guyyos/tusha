from dash import Dash
import dash_bootstrap_components as dbc
import dash_html_components as html


app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP, dbc.icons.BOOTSTRAP])


app.layout = html.Div(
    [
        dbc.Button(className="bi bi-trash  rounded-circle m-4", outline=True, color="primary"),
        dbc.Button(className="bi bi-plus-lg rounded-circle", outline=True, color="primary")
    ],
)

if __name__ == "__main__":
    app.run_server(debug=True)