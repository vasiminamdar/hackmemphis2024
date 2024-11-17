import pandas as pd
import geopandas as gpd
import plotly.express as px
import dash
from dash import Dash, callback, html, dcc, Input, Output, State, no_update
from dash.exceptions import PreventUpdate
import base64
import io
import plotly.graph_objects as go
import pandas as pd
import openai
from dash import dash_table


import api_call_get_data

# sk-proj-

# Load datasets with enhanced handling
open_requests_path = '1_open_requests_2024.csv'
closed_requests_path = '1_closed_requests_2024.csv'
landbank_properties_path = '1_landbank_properties_2024.csv'
blight_neighborhood_dataset_path = '1_blight_neighborhood_dataset_2024.csv'

# Load datasets and handle warnings
open_requests = pd.read_csv(open_requests_path, low_memory=False)
closed_requests = pd.read_csv(closed_requests_path, low_memory=False)
landbank_properties = pd.read_csv(landbank_properties_path, low_memory=False)
blight_neighborhood_dataset = pd.read_csv(blight_neighborhood_dataset_path, low_memory=False)

# Extract latitude and longitude if the 'Location 1' column exists
if "Location 1" in open_requests.columns:
    open_requests[['longitude', 'latitude']] = open_requests['Location 1'].str.extract(r'POINT \(([^ ]+) ([^ ]+)\)')
    open_requests['latitude'] = pd.to_numeric(open_requests['latitude'], errors='coerce')
    open_requests['longitude'] = pd.to_numeric(open_requests['longitude'], errors='coerce')

# Set up the OpenAI API key (replace with your key)

# Summarize datasets
def summarize_dataset(dataset, dataset_name):
    """Summarize a dataset for use in the OpenAI prompt."""
    summary = {
        "Dataset": dataset_name,
        "Number of Rows": len(dataset),
        "Number of Columns": len(dataset.columns),
        "Columns": list(dataset.columns),
        "Sample Data": dataset.head(3).to_dict(orient="records")
    }
    return summary

def format_response(response_text):
    """
    Format the plain text response into structured HTML with bold, newlines, and other elements.
    """
    # Split response into paragraphs for better readability
    paragraphs = response_text.split("\n\n")  # Assuming double newline separates sections

    formatted_response = []
    for para in paragraphs:
        if para.startswith("- "):  # Bullet points
            bullet_points = [
                html.Li(line[2:]) for line in para.split("\n") if line.startswith("- ")
            ]
            formatted_response.append(html.Ul(bullet_points))
        elif para.startswith("#"):  # Markdown-style headers
            header_level = para.count("#")  # Number of '#' defines header level
            header_text = para.replace("#", "").strip()
            if header_level == 1:
                formatted_response.append(html.H1(header_text))
            elif header_level == 2:
                formatted_response.append(html.H2(header_text))
            elif header_level == 3:
                formatted_response.append(html.H3(header_text))
        else:  # Plain text as a paragraph
            formatted_response.append(html.P(para))

    return html.Div(formatted_response)

# Pre-summarize datasets for quick use in chatbot
dataset_summaries = {
    "Open Requests": summarize_dataset(open_requests, "Open Requests"),
    "Closed Requests": summarize_dataset(closed_requests, "Closed Requests"),
    "Landbank Properties": summarize_dataset(landbank_properties, "Landbank Properties"),
    "Blight Neighborhood Dataset": summarize_dataset(blight_neighborhood_dataset, "Blight Neighborhood Dataset")
}

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']


app = dash.Dash(__name__, external_stylesheets=external_stylesheets, suppress_callback_exceptions=True)
server = app.server

# # API URLs
# open_requests_url = "https://data.memphistn.gov/resource/aiee-9zqu.json"
# closed_requests_url = "https://data.memphistn.gov/resource/2244-gnrp.json"

# # Output file paths
# open_requests_csv = "open_requests.parquet"
# closed_requests_csv = "closed_requests.parquet"

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
        ),
        html.Div(
            id='blightchat',
            children=[
            html.H1("City Service Chat", style={"padding-top": "50px", "textAlign": "center", "fontSize": "32px"}),

            # Chatbox Section
            html.Div([
                html.H3("Ask the experts:"),
                dcc.Textarea(
                    id='chatbox-input',
                    placeholder="Type your question here",
                    style={'width': '100%', 'height': 100}
                ),
                html.Button('Submit', id='chatbox-submit', n_clicks=0),
                html.Div(
                    id='chatbox-response',
                    style={'padding': '10px', 'border': '1px solid #ddd', 'margin-top': '10px'}
                )
            ], style={"padding": "20px", "margin": "10px", "borderRadius": "8px"})
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
    print(df.columns)
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
        color="count_crimes",
        color_continuous_scale="Viridis",
        range_color=(merged["count_crimes"].min(), merged["count_crimes"].max()),
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
        df.to_parquet('input_df.parquet', index=None)

        df = df.groupby(['tract']).agg(count=("UCR Category", "count")).reset_index()
        df = df.rename(columns={"count": "count_crimes"})

        merged = geojson_data.merge(df, on="tract")

        fig = go.Figure()

        px_fig = px.choropleth_mapbox(
            merged,
            geojson=geojson_data.__geo_interface__,
            locations="tract",
            featureidkey="properties.tract",
            color='count_crimes',
            color_continuous_scale="Viridis",
            range_color=(merged["count_crimes"].min(), merged["count_crimes"].max()),
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


@app.callback(
    Output("chatbox-response", "children"),
    Input("chatbox-submit", "n_clicks"),
    State("chatbox-input", "value")
)
def chat_response(n_clicks, message):
    if n_clicks > 0 and message:
        # Combine all dataset data into a prompt-friendly format
        dataset_contexts = []
        for dataset_name, dataset in {
            "Open Requests": open_requests,
            "Closed Requests": closed_requests,
            "Landbank Properties": landbank_properties,
            "Blight Neighborhood Dataset": blight_neighborhood_dataset,
        }.items():
            # Sample the first few rows to provide context
            sample_data = dataset.head(50).to_dict(orient="records")
            dataset_contexts.append(
                f"Dataset '{dataset_name}':\n"
                f"- Rows: {len(dataset)}\n"
                f"- Columns: {list(dataset.columns)}\n"
                f"- Sample Data: {sample_data}"
            )
        combined_context = "\n\n".join(dataset_contexts)

        # Construct the prompt with user query and combined dataset context
        prompt = f"""
        You are a data assistant. Here is the context of multiple datasets:
        {combined_context}

        The user has asked the following question:
        {message}

        Please provide a detailed, conversational response using the data provided. If specific analysis or insights are required, interpret the data as needed.
        """

        try:
            response = openai.ChatCompletion.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1000,  # Increase token limit for more detailed responses
                temperature=0.7  # Balance creativity with precision
            )

            # Parse and format the response
            response_text = response['choices'][0]['message']['content']
            return format_response(response_text)

        except Exception as e:
            return f"Oops, something went wrong: {str(e)}"

    raise PreventUpdate

if __name__ == "__main__":
    app.run_server(debug=True)