import pandas as pd
import geopandas as gpd
import plotly.express as px
import dash
from dash import Dash, callback, html, dcc, Input, Output, State, no_update
from dash.exceptions import PreventUpdate
import base64
import io
import plotly.graph_objects as go

import api_call_get_data


external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']


app = dash.Dash(__name__, external_stylesheets=external_stylesheets, suppress_callback_exceptions=True)
server = app.server

# API URLs
open_requests_url = "https://data.memphistn.gov/resource/aiee-9zqu.json"
closed_requests_url = "https://data.memphistn.gov/resource/2244-gnrp.json"

# Output file paths
open_requests_csv = "open_requests.parquet"
closed_requests_csv = "closed_requests.parquet"

# Fetch and save open and closed requests
# api_call_get_data.fetch_data(open_requests_url, limit=50000, output_file=open_requests_csv)
# api_call_get_data.fetch_data(closed_requests_url, limit=50000, output_file=closed_requests_csv)

# # Load GeoJSON file for Memphis ZIP codes
# geojson_file = "memphis_tract.geojson"
# geojson_data = gpd.read_file(geojson_file)

# # Load CSV data for homes by ZIP code
# data = pd.read_csv('d.csv')
# crime = pd.read_csv('crime_agg_tract.csv')
# merged = geojson_data.merge(data, on="tract")

# merged['boundary'] = [0] * data.shape[0]


app.layout = html.Div(
    id="main",
    children=[
        html.H1(
            "Memphis Blight Data Dashboard",
            style={
                "display": "flex",
                "justifyContent": "center",
                "alignItems": "center",
                "height": "30px",
                "padding-top": "10px"
            }
        ),
        html.Br(),
        html.Div(
            dcc.Dropdown(
                id='dropdown',
                options=['census tract', 'zipcode'],
                value='census tract',
                placeholder='Choose a map layer',
                style={"display": "none"}
            ),
        ),
        html.Div(
            id="maps",
            children=
            [
            dcc.Upload(
                id="file-upload",
                children=[
                    html.Button('Upload File')
                    ],
                    style={
                        "display": "flex",
                        "justifyContent": "center",
                        "alignItems": "center",
                        "height": "30px"
                    }
                        ),
            html.Div(id="output-data-upload")
        ]
        )
    ]
)

@callback(
        Output('maps', 'children', allow_duplicate=True),
        Input('dropdown', 'value'),
        prevent_initial_call=True
)
def update_output(layer):

    data = pd.read_csv('blight.csv').iloc[:5000]

    geojson_file = "memphis_tract.geojson" if layer == 'census tract' else "memphis.geojson"
    field = 'tract' if layer == 'census tract' else 'zip'
    geojson_data = gpd.read_file(geojson_file)

    df = pd.read_parquet('input_df.parquet')
    df[field] = df[field].astype('int32')

    df = df.groupby([field]).agg(count=("UCR Category", "count")).reset_index()
    df = df.rename(columns={"count": "count_crimes"})

    merged = geojson_data.merge(df, on=field)

    fig = go.Figure()

    px_fig = px.choropleth_mapbox(
        merged,
        geojson=geojson_data.__geo_interface__,
        locations=field,
        featureidkey=f"properties.{field}",
        color=field,
        color_continuous_scale="Viridis",
        range_color=(merged[field].min(), merged[field].max()),
        mapbox_style="carto-positron",
        hover_data={field: True},
        zoom=10,
        center={"lat": 35.1495, "lon": -90.0490},  # Coordinates for Memphis, TN
        opacity=0.2
    )

    fig.add_traces(px_fig.data)
    fig.update_layout(px_fig.layout)

    data["hover_text"] = (
        "Address: " + data["address"]
    )

    data1 = dict(
        type='scattermapbox',
        lat=data['latitude'],
        lon=data['longitude'],
        marker=dict(color='blue'),
        name='Plot 1',
        hoverinfo='text',
        text=data['hover_text']
    )

    fig.add_trace(data1)

    # Create the layout for the figure
    layout = go.Layout(
        mapbox=dict(
            style="carto-positron",
            center=dict(lat=35.1495, lon=-90.0490),  # Center the map
            zoom=10
        ),
        title_x=0.5,
        legend_title="Legend",
        height=700,
        font=dict(
            family="Arial, sans-serif",  # Font family
            size=16
        ),
        title="Map with Initial Scattermapbox Layer"
    )

    fig.update_layout(layout)

    figure = dcc.Graph(
            figure=fig,
            style={
                "padding-top": "20px"
            }
    )

    return figure


@callback(
        Output('dropdown', 'style'),
        Output('maps', 'children'),
        Input('file-upload', 'contents'),
        State('dropdown', 'style')
)
def create_map(contents, dropdown):

    if contents:

        if dropdown.get('display'):
            del dropdown['display']

        data = pd.read_csv('blight.csv').iloc[:5000]

        geojson_file = "memphis_tract.geojson"
        geojson_data = gpd.read_file(geojson_file)

        _, content_string = contents.split(",")
        decoded = base64.b64decode(content_string)

        df = pd.read_parquet(io.BytesIO(decoded))
        df['tract'] = df['tract'].astype('int32')
        df = df.groupby(['tract']).agg(count=("UCR Category", "count")).reset_index()
        df = df.rename(columns={"count": "count_crimes"})

        merged = geojson_data.merge(df, on="tract")

        df.to_parquet('input_df.parquet', index=None)

        fig = go.Figure()

        px_fig = px.choropleth_mapbox(
            merged,
            geojson=geojson_data.__geo_interface__,
            locations="tract",
            featureidkey="properties.tract",
            color='count',
            color_continuous_scale="Viridis",
            range_color=(merged["count"].min(), merged["count"].max()),
            mapbox_style="carto-positron",
            hover_data={'tract': True},
            zoom=10,
            center={"lat": 35.1495, "lon": -90.0490},  # Coordinates for Memphis, TN
            opacity=0.2
        )

        fig.add_traces(px_fig.data)
        fig.update_layout(px_fig.layout)

        data["hover_text"] = (
        "Address: " + data["address"]
        )

        data1 = dict(
            type='scattermapbox',
            lat=data['latitude'],
            lon=data['longitude'],
            marker=dict(color='blue'),
            name='Plot 1',
            hoverinfo='text',
            text=data['hover_text']

        )

        fig.add_trace(data1)

        # Create the layout for the figure
        layout = go.Layout(
            mapbox=dict(
                style="carto-positron",
                center=dict(lat=35.1495, lon=-90.0490),  # Center the map
                zoom=10
            ),
            title_x=0.5,
            legend_title="Legend",
            height=700,
            font=dict(
                family="Arial, sans-serif",  # Font family
                size=16
            ),
            title="Map with Initial Scattermapbox Layer"
        )

        fig.update_layout(layout)


        figure = dcc.Graph(
                figure=fig,
                style={
                    "padding-top": "20px"
                }
        )

        return dropdown, figure

    return no_update

if __name__ == "__main__":
    app.run_server(debug=True)