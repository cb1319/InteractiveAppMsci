from dash import dcc, html, Input, Output, no_update, Dash
import plotly.graph_objects as go
import pandas as pd
import pathlib
import dash_bootstrap_components as dbc

this_dir = pathlib.Path(__file__).parent
feature_path = this_dir.parent / 'data/interactive_results.pkl'

print(f'Loading data from {feature_path}')

morgan_keys = pd.read_pickle(feature_path)

print(f'Loaded data with shape {morgan_keys.shape}')

# Define list of y-axis targets and models
targets = ['IP', 'ES1', 'fosc1']
models = ['average', 'SchNet', 'DimeNet', 'DimeNetPlusPlus', 'PaiNN', 'Equiformer', 'SphereNet']
x_axis = ['CT', 'MW', 'Similarity_Score', 'Cluster']

# Create PCA scatter plot
pca_fig = go.Figure(data=[
    go.Scatter(
        x=morgan_keys['pca1'],
        y=morgan_keys['pca2'],
        mode="markers",
        marker=dict(
            color=morgan_keys['Cluster'],  # Color by Cluster number
            colorscale='viridis',  # Use viridis color scale
            size=10,
            colorbar={"title": "Cluster"},
            line={"color": "#444"},
            opacity=0.8,
        ),
    )
])

pca_fig.update_layout(
    xaxis=dict(title='PCA1'),
    yaxis=dict(title='PCA2'),
    plot_bgcolor='rgba(255,255,255,0.1)',
    width=800,
    height=400,
    title='Reduced dimensionality visualisation of fragments using PCA'
)

colors = {
    'background': '#E6E6FA',
    'text': '#000000'
}

# Initialize Dash app
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server

# Define app layout
app.layout = html.Div([
    html.Div([
        dbc.Row([
            html.Div(style={'margin': '0', 'padding': '0', 'overflowX': 'hidden', 'justify-content': 'center'}, children=[
                html.H1(children='MSci Project 2023-2024 Cyprien Bone - Interactive Dashboard',
                    style={'textAlign': 'center', 'color': colors['text'], 'fontFamily': 'Arial'}),
                html.H2(children='Fragments Scaffold Explorer',
                    style={'textAlign': 'center', 'color': colors['text'], 'fontFamily': 'Arial'}),
                html.P(children='This dashboard allows you to visualise the effect of different models, targets, and properties on GNN predictions for organic semiconductor fragments.',
                    style={'textAlign': 'center', 'color': colors['text'], 'fontFamily': 'Arial'}),
            ], className='w-100')  # Set full width for the row and center horizontally
        ], justify='center'),  # Center horizontally
        dbc.Row([
            dbc.Col(
                # put the plot in the middle of the page
                dcc.Graph(id="pca-graph", figure=pca_fig, clear_on_unhover=True),
                width={'size': 10}  # Set width to 10 out of 12 columns
            )
        ], justify='center'),  # Center horizontally
        dbc.Row([
            html.H2(children='Model Performance Comparison (tick buttons to show/hide models)',
                style={'textAlign': 'center', 'color': colors['text'], 'fontFamily': 'Arial'}),
        ], justify='center'),  # Center horizontally
        dbc.Row([
            dbc.Col(
                html.Div([
                    dcc.Graph(id="graph", clear_on_unhover=True),
                    dcc.Tooltip(id="graph-tooltip"),
                    dcc.Dropdown(
                        id='y-axis-target-dropdown',
                        options=[{'label': target, 'value': target} for target in targets],
                        value=targets[0],
                        clearable=False,
                    ),
                    dcc.Dropdown(
                        id='x-axis-dropdown',
                        options=[{'label': x, 'value': x} for x in x_axis],
                        value=x_axis[0],
                        clearable=False,
                    ),
                    html.Div([
                        html.Button(id=f'y-axis-model-button-{model}', n_clicks=0, children=model, style={'margin-right': '10px'})
                        for model in models
                    ]),
                ]),
                width={'size': 10}  # Set width to 10 out of 12 columns
            )
        ], justify='center'),  # Center horizontally
    ], style={'min-height': '90vh'}),  # Adjust the height as per your requirement
    html.Footer(
        "MSci Project 2023-2024 - Imperial College London, generated with the help of the Jelfs group, render, dash and ChatGPT",
        style={'padding': '100px', 'background-color': '#f8f9fa', 'text-align': 'center', 'font-family': 'Arial', 'font-size': '15px'}
    )
])

# Define callback to update the graph
@app.callback(
    Output("graph", "figure"),
    Output("graph-tooltip", "show"),
    Output("graph-tooltip", "bbox"),
    Output("graph-tooltip", "children"),
    Input("graph", "hoverData"),
    Input("pca-graph", "hoverData"),
    Input("x-axis-dropdown", "value"),
    Input("y-axis-target-dropdown", "value"),
    *[Input(f'y-axis-model-button-{model}', 'n_clicks') for model in models],
)
def update_graph(graph_hoverData, pca_hoverData, x_axis, y_target, *n_clicks):
    # Get selected models
    selected_models = [model for model, clicks in zip(models, n_clicks) if clicks % 2 != 0]

    # Filter Morgan keys data based on selected x_axis and y_target
    filtered_data = morgan_keys[[x_axis] + [f'mean_diff_{model}_{y_target}_weighted' for model in selected_models]]

    # Create scatter plot
    fig = go.Figure()
    for model in selected_models:
        y_column = f'mean_diff_{model}_{y_target}_weighted'
        fig.add_trace(go.Scatter(
            x=filtered_data[x_axis],
            y=filtered_data[y_column],
            mode="markers",
            marker=dict(
                color=morgan_keys['Cluster'], 
                colorscale='viridis', 
                colorbar={"title": "Cluster"}, 
                size=10,
                line={"color": "#444"},
                opacity=0.8,
            ),
            name=y_column
        ))

    fig.update_layout(
        xaxis=dict(title=f'{x_axis}'),
        yaxis=dict(title=f'MAE_{y_target}'),
        plot_bgcolor='rgba(255,255,255,0.1)',
        width=800,
        height=400,
        title='Effects of models targets and properties on errors associated to each fragment'
    )

    # Determine if tooltip should be shown
    if graph_hoverData is None:
        show_tooltip = False
        bbox = no_update
        children = no_update
    
    if pca_hoverData is None:
        show_tooltip = False
        bbox = no_update
        children = no_update

    if graph_hoverData is not None:
        # Demo only shows the first point, but other points may also be available
        pt = graph_hoverData["points"][0]
        bbox = pt["bbox"]
        num = pt["pointNumber"]
        im_url = morgan_keys['images_url'][num]

        children = [
            html.Div(children=[
                html.Img(src=im_url, style={"width": "100%"}),
            ],
            style={'width': '200px', 'white-space': 'normal'})
        ]

        show_tooltip = True

    if pca_hoverData is not None:
        # Demo only shows the first point, but other points may also be available
        pt = pca_hoverData["points"][0]
        bbox = pt["bbox"]
        num = pt["pointNumber"]
        im_url = morgan_keys['images_url'][num]

        children = [
            html.Div(children=[
                html.Img(src=im_url, style={"width": "100%"}),
            ],
            style={'width': '200px', 'white-space': 'normal'})
        ]

        show_tooltip = True

    return fig, show_tooltip, bbox, children

if __name__ == "__main__":
    app.run_server(debug=True)
