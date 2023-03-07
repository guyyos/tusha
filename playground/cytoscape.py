from dash import Dash, html
import dash_cytoscape as cyto

app = Dash(__name__)

directed_edges = [
    {'data': {'id': src+tgt, 'source': src, 'target': tgt}}
    for src, tgt in ['BA', 'BC', 'CD', 'DA']
]

directed_elements = [{'data': {'id': id_}} for id_ in 'ABCD'] + directed_edges

graph1 = cyto.Cytoscape(
        id='cytoscape-styling-9',
        layout={'name': 'circle'},
        style={'width': '100%', 'height': '400px'},
        elements=directed_elements,
        stylesheet=[
            {
                'selector': 'node',
                'style': {
                    'label': 'data(id)'
                }
            },
            {
                'selector': 'edge',
                'style': {
                    # The default curve style does not work with certain arrows
                    'curve-style': 'bezier'
                }
            },
            {
                'selector': '#BA',
                'style': {
                    'source-arrow-color': 'red',
                    'source-arrow-shape': 'triangle',
                    'line-color': 'red'
                }
            },
            {
                'selector': '#DA',
                'style': {
                    'target-arrow-color': 'blue',
                    'target-arrow-shape': 'vee',
                    'line-color': 'blue'
                }
            },
            {
                'selector': '#BC',
                'style': {
                    'mid-source-arrow-color': 'green',
                    'mid-source-arrow-shape': 'diamond',
                    'mid-source-arrow-fill': 'hollow',
                    'line-color': 'green',
                }
            },
            {
                'selector': '#CD',
                'style': {
                    'mid-target-arrow-color': 'black',
                    'mid-target-arrow-shape': 'circle',
                    'arrow-scale': 2,
                    'line-color': 'black',
                }
            }
        ]
    )


graph2 = cyto.Cytoscape(
        id='cytoscape-compound',
        layout={'name': 'preset'},
        style={'width': '100%', 'height': '450px'},
        stylesheet=[
    
            {
                'selector': 'node',
                'style': {'content': 'data(label)'}
            },
            {
                'selector': '.countries',
                # 'style': {'width': 5},
                'style': {'width': 1,
                    'target-arrow-color': 'blue',
                    'target-arrow-shape': 'vee',
                    'line-color': 'blue'
                }
            },
            {
                'selector': '.cities',
                'style': {'line-style': 'dashed'}
            }
        ],
        elements=[
            # Parent Nodes
            {
                'data': {'id': 'us', 'label': 'United States'}
            },
            {
                'data': {'id': 'can', 'label': 'Canada'}
            },

            # Children Nodes
            {
                'data': {'id': 'nyc', 'label': 'New York', 'parent': 'us'},
                'position': {'x': 100, 'y': 100}
            },
            {
                'data': {'id': 'sf', 'label': 'San Francisco', 'parent': 'us'},
                'position': {'x': 100, 'y': 200}
            },
            {
                'data': {'id': 'mtl', 'label': 'Montreal', 'parent': 'can'},
                'position': {'x': 400, 'y': 100}
            },

            # Edges
            {
                'data': {'source': 'can', 'target': 'us'},
                'classes': 'countries'
            },
            {
                'data': {'source': 'nyc', 'target': 'sf'},
                'classes': 'cities'
            },
            {
                'data': {'source': 'sf', 'target': 'mtl'},
                'classes': 'cities'
            }
        ]
    )

import pandas as pd
from pandas import DataFrame 

df_relations = DataFrame(columns = ['Cause','Effect'],data=[['cause1','effect1'],['cause1','effect2']])

directed_edges = df_relations.apply(lambda x:{'data': {'id': x['Cause']+x['Effect'], 'source': x['Cause'], 'target': x['Effect']}},axis=1)

nodes = set(df_relations.Cause.unique())|set(df_relations.Effect.unique())

directed_elements = [{'data': {'id': id_}} for id_ in nodes] + list(directed_edges)


graph3 = cyto.Cytoscape(
        id='graph1',
        layout={'name': 'breadthfirst'},#cose  breadthfirst
        style={'width': '30%', 'height': '250px'},

        stylesheet=[
    
            {
                'selector': 'node',
                'style': {
                    'label': 'data(id)'
                }
            },
            {
                'selector': 'edge',
                'style': {
                    'curve-style': 'bezier',
                    'target-arrow-color': 'blue',
                    'target-arrow-shape': 'vee',
                    'line-color': 'blue'
                }
            }
        ],
        elements=directed_elements
    )


app.layout = html.Div([graph1,graph3])


if __name__ == '__main__':
    app.run_server(debug=True,port=8051)
