import pandas as pd
import geopandas as gpd
import plotly.express as px
import dash
from dash import Dash, callback, html, dcc, Input, Output, State, no_update
from dash.exceptions import PreventUpdate
import base64
import io
import plotly.graph_objects as go
import openai
from dash import dash_table
from dash import html


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
    Format the plain text response into structured HTML.
    Make "Conclusion" and "Result" bold while keeping the rest of the content normal.
    """
    # Split response into paragraphs for better readability
    paragraphs = response_text.split("\n\n")  # Assuming double newline separates sections

    formatted_response = []
    for para in paragraphs:
        # Check for specific keywords to bold
        if "Conclusion" in para or "Result" in para:
            # Split the paragraph into parts and replace keywords with bold elements
            parts = []
            for word in para.split():
                if word == "Conclusion":
                    parts.append(html.B("Conclusion"))
                elif word == "Result":
                    parts.append(html.B("Result"))
                else:
                    parts.append(f"{word} ")
            formatted_response.append(html.P(parts))
        else:
            formatted_response.append(html.P(para))

    return html.Div(formatted_response)

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']


app = dash.Dash(__name__, external_stylesheets=external_stylesheets, suppress_callback_exceptions=True)
server = app.server

data = pd.read_csv('blight.csv').iloc[:5000]

data["hover_text"] = (
    "Address: " + data["address"]
)

geojson_file = "memphis_tract.geojson"
geojson_data = gpd.read_file(geojson_file)

merged = pd.DataFrame()
merged['tract'] = geojson_data['tract']

fig = go.Figure()

px_fig = px.choropleth_mapbox(
    merged,
    geojson=geojson_data.__geo_interface__,
    locations='tract',
    featureidkey=f"properties.tract",
    color_continuous_scale="Viridis",
    range_color=(0,0),
    mapbox_style="carto-positron",
    hover_data={'tract': True},
    zoom=10,
    center={"lat": 35.1495, "lon": -90.0490},  # Coordinates for Memphis, TN
    opacity=0.2
)

fig.add_traces(px_fig.data)
fig.update_layout(px_fig.layout)

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
    title="Blight data in Memphis"
)

fig.update_layout(layout)
fig.update_traces(showlegend=False)

 # Set up the OpenAI API key (replace with your key)
# Combine all dataset data


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
            children=[
            dcc.Dropdown(
                id='dropdown',
                options=['census tract', 'zipcode'],
                value='census tract',
                placeholder='Choose a map layer',
                style={"display": "none", "width": "50%", "padding-left": "250px"}
            ),
            dcc.Dropdown(
                id='agg',
                options=[],
                value='',
                placeholder="Choose a variable to aggregate the data",
                style={"display": "none", "width": "50%"}
            ),
            ],
            style={
                "display": "flex",
                "justify-content": "center",
                "align-items": "center"
            }
        ),
        html.Div(
            id="maps",
            children=
            [
            dcc.Upload(
                id="file-upload",
                children=[
                    html.Button('Upload Your Data')
                    ],
                    style={
                        "display": "flex",
                        "justifyContent": "center",
                        "alignItems": "center",
                        "height": "30px"
                    }
            ),
            html.H6(
                "*Your data file must include latitude and longitude columns*",
                 style={
                    "display": "flex",
                    "justifyContent": "center",
                    "alignItems": "center"
                }
            ),
            dcc.Graph(
                id="start",
                figure=fig,
                style={
                    "padding-top": "20px"
                }
            )
            ]
            ),
        html.Div(
            id='blightchat',
            children=[
            # html.H1("City Service Chat", style={"padding-top": "50px", "textAlign": "center", "fontSize": "32px"}),

            # Chatbox Section
            html.Div([
                html.H3("Explore and query the data with AI"),
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
            ], style={ "borderRadius": "8px"})
            ]
        )
    ]
)

@callback(
        Output('maps', 'children', allow_duplicate=True),
        Input('dropdown', 'value'),
        Input('agg', 'value'),
        prevent_initial_call=True
)
def update_output(layer, aggval):

    data = pd.read_csv('blight.csv').iloc[:5000]

    geojson_file = "memphis_tract.geojson" if layer == 'census tract' else "memphis.geojson"
    field = 'tract' if layer == 'census tract' else 'zip'
    geojson_data = gpd.read_file(geojson_file)

    df = pd.read_parquet('input_df.parquet')

    df[field] = df[field].astype('int32')

    if not aggval:

        aggval = "crimes"

        temp = df.groupby(field).agg(count=('UCR Category', "count")).reset_index()
        df = temp.rename(columns={"count": f"count_{aggval}"})

    else:
        
        df = df.loc[df['UCR Category'] == aggval]
        temp = df.groupby(field).agg(count=('UCR Category', "count")).reset_index()
        df = temp.rename(columns={"count": f"count_{aggval}"})

    merged = geojson_data.merge(df, on=field)
    # merged[f"count_{aggval}"] = merged[f"count_{aggval}"].fillna(0)

    fig = go.Figure()

    px_fig = px.choropleth_mapbox(
        merged,
        geojson=geojson_data.__geo_interface__,
        locations=field,
        featureidkey=f"properties.{field}",
        color=f"count_{aggval}",
        color_continuous_scale="Viridis",
        range_color=(merged[f"count_{aggval}"].min(), merged[f"count_{aggval}"].max()),
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
        title="Blight data in Memphis with aggregated crime statistics map overlay"
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
        Output('agg', 'style'),
        Output('agg', 'options'),
        Output('dropdown', 'style'),
        Output('maps', 'children'),
        Input('file-upload', 'contents'),
        State('dropdown', 'style'),
        State('agg', 'options'),
        State('agg', 'style')
)
def create_map(contents, dropdown, options, aggstyle):

    if contents:

        if dropdown.get('display'):
            del dropdown['display']
        if aggstyle.get('display'):
            del aggstyle['display']

        data = pd.read_csv('blight.csv').iloc[:5000]

        geojson_file = "memphis_tract.geojson"
        geojson_data = gpd.read_file(geojson_file)

        _, content_string = contents.split(",")
        decoded = base64.b64decode(content_string)

        df = pd.read_parquet(io.BytesIO(decoded))
        df.to_parquet('input_df.parquet', index=None)

        options = [i for i in df['UCR Category'].unique() if i]

        df['tract'] = df['tract'].astype('int32')

        temp = df.groupby('tract').agg(count=('UCR Category', "count")).reset_index()
        df = temp.rename(columns={"count": "count_crimes"})

        merged = geojson_data.merge(df, on="tract")
        # merged['count_crimes'] = merged['count_crimes'].fillna(0)

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
            title="Blight data in Memphis with aggregated crime statistics map overlay"
        )

        fig.update_layout(layout)

        figure = dcc.Graph(
                figure=fig,
                style={
                    "padding-top": "20px"
                }
        )

        return aggstyle, options, dropdown, figure

    return no_update


@app.callback(
    Output("chatbox-response", "children"),
    Input("chatbox-submit", "n_clicks"),
    State("chatbox-input", "value")
)
def chat_response(n_clicks, message):
    if n_clicks > 0 and message:

        # Load datasets with enhanced handling
        open_requests_path = '1_open_requests_2024.csv'
        closed_requests_path = '1_closed_requests_2024.csv'
        landbank_properties_path = '1_landbank_properties_2024.csv'
        blight_neighborhood_dataset_path = '1_blight_neighborhood_dataset_2024.csv'
        crime_dataset_path = '1_crime_data.csv'

        # Load datasets and handle warnings
        open_requests = pd.read_csv(open_requests_path, low_memory=False)
        closed_requests = pd.read_csv(closed_requests_path, low_memory=False)
        landbank_properties = pd.read_csv(landbank_properties_path, low_memory=False)
        blight_neighborhood_dataset = pd.read_csv(blight_neighborhood_dataset_path, low_memory=False)
        crime_dataset = pd.read_csv(crime_dataset_path, low_memory=False)

        # Pre-summarize datasets for quick use in chatbot
        dataset_summaries = {
            "Open Requests": summarize_dataset(open_requests, "Open Requests"),
            "Closed Requests": summarize_dataset(closed_requests, "Closed Requests"),
            "Landbank Properties": summarize_dataset(landbank_properties, "Landbank Properties"),
            "Blight Neighborhood Dataset": summarize_dataset(blight_neighborhood_dataset, "Blight Neighborhood Dataset"),
            "Crime Dataset": summarize_dataset(crime_dataset, "Crime Dataset")
        }

        # Extract latitude and longitude if the 'Location 1' column exists
        if "Location 1" in open_requests.columns:
            open_requests[['longitude', 'latitude']] = open_requests['Location 1'].str.extract(r'POINT \(([^ ]+) ([^ ]+)\)')
            open_requests['latitude'] = pd.to_numeric(open_requests['latitude'], errors='coerce')
            open_requests['longitude'] = pd.to_numeric(open_requests['longitude'], errors='coerce')
        
        dataset_contexts = []
        for dataset_name, dataset in {
            "Open Requests": open_requests,
            "Closed Requests": closed_requests,
            "Landbank Properties": landbank_properties,
            "Blight Neighborhood Dataset": blight_neighborhood_dataset,
            "Crime Dataset": crime_dataset
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