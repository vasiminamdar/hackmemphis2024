import pandas as pd
import dash
from dash import dcc, html, Input, Output
import plotly.express as px

# Load data
data_path = '/Users/vasiminamdar/Downloads/blight/Citizen_Connect_Open_Public_Works_Service_Requests_20241113.csv'
data = pd.read_csv(data_path)


# Extract latitude and longitude from 'Location 1' in the format 'POINT (longitude latitude)'
if "Location 1" in data.columns:
    data[['longitude', 'latitude']] = data['Location 1'].str.extract(r'POINT \(([^ ]+) ([^ ]+)\)')
    data['latitude'] = pd.to_numeric(data['latitude'], errors='coerce')
    data['longitude'] = pd.to_numeric(data['longitude'], errors='coerce')

# Initialize the Dash app
app = dash.Dash(__name__)

# Define color schemes and styles
CHART_COLOR = "#2c3e50"
BG_COLOR = "#ecf0f1"
TEXT_COLOR = "#34495e"

app.layout = html.Div([
    html.H1("City Service and Community Insights Dashboard", style={"textAlign": "center", "color": TEXT_COLOR, "fontSize": "32px"}),
    html.Button(id='dummy', n_clicks=0, style={'display': 'none'}),  # Dummy input for triggering callbacks

    # Sections for each dashboard with background styling
    html.Div([html.H2("Identifying Geographic Areas for More Frequent City Service"), dcc.Graph(id="service-areas-chart")],
             style={"backgroundColor": BG_COLOR, "padding": "20px", "margin": "10px", "borderRadius": "8px"}),
    html.Div([html.H2("Service Request by Shelby County Zip Code"), dcc.Graph(id="service-requests-zip-chart")],
             style={"backgroundColor": BG_COLOR, "padding": "20px", "margin": "10px", "borderRadius": "8px"}),
    html.Div([html.H2("Service Request Frequency by Category"), dcc.Graph(id="service-frequency-category-chart")],
             style={"backgroundColor": BG_COLOR, "padding": "20px", "margin": "10px", "borderRadius": "8px"}),
    html.Div([html.H2("Resident Education & Outreach Strategy"), dcc.Graph(id="education-outreach-strategy-chart")],
             style={"backgroundColor": BG_COLOR, "padding": "20px", "margin": "10px", "borderRadius": "8px"}),
    html.Div([html.H2("Recurring Resident Feedback"), dcc.Graph(id="recurring-feedback-chart")],
             style={"backgroundColor": BG_COLOR, "padding": "20px", "margin": "10px", "borderRadius": "8px"}),
    html.Div([html.H2("Service Frequency & Geographic Areas"), dcc.Graph(id="service-frequency-chart")],
             style={"backgroundColor": BG_COLOR, "padding": "20px", "margin": "10px", "borderRadius": "8px"}),
    html.Div([html.H2("Blight Predictive Indicators (Sorted by Count)"), dcc.Graph(id="blight-indicators-chart")],
             style={"backgroundColor": BG_COLOR, "padding": "20px", "margin": "10px", "borderRadius": "8px"}),
    html.Div([html.H2("311 Data Quality (Columns with Missing Values)"), dcc.Graph(id="data-quality-chart")],
             style={"backgroundColor": BG_COLOR, "padding": "20px", "margin": "10px", "borderRadius": "8px"}),
    html.Div([html.H2("Monthly Trend of Service Requests"), dcc.Graph(id="monthly-trend-chart")],
             style={"backgroundColor": BG_COLOR, "padding": "20px", "margin": "10px", "borderRadius": "8px"}),
    html.Div([html.H2("Top 10 Most Common Request Types by Sub-District"), dcc.Graph(id="top-requests-subdistrict-chart")],
             style={"backgroundColor": BG_COLOR, "padding": "20px", "margin": "10px", "borderRadius": "8px"}),
    html.Div([html.H2("Service Requests by Day of the Week"), dcc.Graph(id="day-of-week-chart")],
             style={"backgroundColor": BG_COLOR, "padding": "20px", "margin": "10px", "borderRadius": "8px"}),
    html.Div([html.H2("Service Requests Heatmap by Hour and Day"), dcc.Graph(id="hour-day-heatmap")],
             style={"backgroundColor": BG_COLOR, "padding": "20px", "margin": "10px", "borderRadius": "8px"}),
    html.Div([html.H2("Service Requests by Month"), dcc.Graph(id="service-requests-month-chart")],
            style={"backgroundColor": BG_COLOR, "padding": "20px", "margin": "10px", "borderRadius": "8px"}),
], style={"backgroundColor": "#f5f6fa", "padding": "20px", "fontFamily": "Arial, sans-serif"})

# Helper function to generate bar and pie charts
def generate_chart(dataframe, chart_type="bar", x=None, y=None, title="", names=None, values=None, color=CHART_COLOR):
    if chart_type == "bar":
        fig = px.bar(dataframe, x=x, y=y, color_discrete_sequence=[color], title=title)
        fig.update_layout(
            title_x=0.5, title_font_size=18, plot_bgcolor=BG_COLOR,
            font=dict(color=TEXT_COLOR), margin=dict(l=20, r=20, t=40, b=20)
        )
    elif chart_type == "pie":
        fig = px.pie(dataframe, names=names, values=values, color_discrete_sequence=[color], title=title)
        fig.update_layout(
            title_x=0.5, title_font_size=18, font=dict(color=TEXT_COLOR),
            margin=dict(l=20, r=20, t=40, b=20)
        )
    return fig

# Callback for each chart
@app.callback(
    Output("service-areas-chart", "figure"),
    Input("dummy", "n_clicks")
)
def update_service_areas_chart(n_clicks):
    data_aggregated = data.groupby(['latitude', 'longitude']).size().reset_index(name='request_count')
    data_aggregated = data_aggregated.dropna(subset=['latitude', 'longitude'])
    fig = px.density_mapbox(
        data_aggregated, lat='latitude', lon='longitude', z='request_count',
        radius=10, center=dict(lat=data_aggregated['latitude'].mean(), lon=data_aggregated['longitude'].mean()),
        color_continuous_scale="Viridis", title="Identifying Geographic Areas for More Frequent City Service"
    )
    fig.update_layout(mapbox_style="carto-positron", margin=dict(l=20, r=20, t=40, b=20), title_font_size=18)
    return fig

# Updated callback to use 'POSTAL_CODE' as the zip code column
@app.callback(
    Output("service-requests-zip-chart", "figure"),
    Input("dummy", "n_clicks")
)
def update_service_requests_zip_chart(n_clicks):
    zip_data = data['POSTAL_CODE'].value_counts().nlargest(25).reset_index()
    zip_data.columns = ['Zip Code', 'Service Requests']
    zip_data = zip_data.sort_values(by='Service Requests', ascending=False)
    return generate_chart(zip_data, chart_type="bar", x='Zip Code', y='Service Requests', title="Service Request by Shelby County Zip Code")

@app.callback(
    Output("service-frequency-category-chart", "figure"),
    Input("dummy", "n_clicks")
)
def update_service_frequency_category_chart(n_clicks):
    category_data = data['CATEGORY'].value_counts().reset_index()
    category_data.columns = ['Category', 'Service Requests']
    return generate_chart(category_data, chart_type="bar", x='Category', y='Service Requests', title="Service Request Frequency by Category")

@app.callback(
    Output("education-outreach-strategy-chart", "figure"),
    Input("dummy", "n_clicks")
)
def update_education_outreach_strategy_chart(n_clicks):
    category_data = data['CATEGORY'].value_counts().reset_index()
    category_data.columns = ['Category', 'Service Requests']
    return generate_chart(category_data, chart_type="pie", names='Category', values='Service Requests', title="Resident Education & Outreach Strategy")

@app.callback(
    Output("recurring-feedback-chart", "figure"),
    Input("dummy", "n_clicks")
)
def update_recurring_feedback_chart(n_clicks):
    top_requests = data['REQUEST_TYPE'].value_counts().nlargest(10).reset_index()
    top_requests.columns = ['Request Type', 'Frequency']
    return generate_chart(top_requests, chart_type="bar", x='Request Type', y='Frequency', title="Top 10 Recurring Service Request Types")

@app.callback(
    Output("service-frequency-chart", "figure"),
    Input("dummy", "n_clicks")
)
def update_service_frequency_chart(n_clicks):
    fig = px.scatter_mapbox(
        data.dropna(subset=['latitude', 'longitude']), lat="latitude", lon="longitude",
        color_discrete_sequence=[CHART_COLOR], hover_data=["REQUEST_TYPE", "INCIDENT_ID"],
        title="Geographic Areas Needing Increased Service Frequency"
    )
    fig.update_layout(mapbox_style="carto-positron", margin=dict(l=20, r=20, t=40, b=20), title_font_size=18)
    return fig

@app.callback(
    Output("blight-indicators-chart", "figure"),
    Input("dummy", "n_clicks")
)
def update_blight_indicators_chart(n_clicks):
    blight_data = data['CATEGORY'].value_counts().reset_index()
    blight_data.columns = ['CATEGORY', 'count']
    blight_data = blight_data.sort_values(by='count', ascending=False)
    return generate_chart(blight_data, chart_type="bar", x='CATEGORY', y='count', title="Blight Predictive Indicators (Sorted by Count)")

@app.callback(
    Output("data-quality-chart", "figure"),
    Input("dummy", "n_clicks")
)
def update_data_quality_chart(n_clicks):
    missing_data = data.isnull().sum().reset_index()
    missing_data.columns = ['Column', 'Missing Values']
    missing_data = missing_data[missing_data['Missing Values'] > 0].sort_values(by='Missing Values', ascending=False)
    return generate_chart(missing_data, chart_type="bar", x='Column', y='Missing Values', title="311 Data Quality (Columns with Missing Values)")

@app.callback(
    Output("monthly-trend-chart", "figure"),
    Input("dummy", "n_clicks")
)
def update_monthly_trend_chart(n_clicks):
    # Convert 'CREATION_DATE' to datetime and then to period, followed by conversion to string
    data['month_year'] = pd.to_datetime(data['CREATION_DATE']).dt.to_period("M").astype(str)
    
    # Get the count of service requests per month
    monthly_data = data['month_year'].value_counts().sort_index().reset_index()
    monthly_data.columns = ['Month', 'Service Requests']
    
    # Generate the line chart
    fig = px.line(monthly_data, x='Month', y='Service Requests', title="Monthly Trend of Service Requests")
    
    # Update layout to ensure proper rendering
    fig.update_layout(margin=dict(l=20, r=20, t=40, b=20))
    
    return fig


@app.callback(
    Output("top-requests-subdistrict-chart", "figure"),
    Input("dummy", "n_clicks")
)
def update_top_requests_subdistrict_chart(n_clicks):
    top_requests_data = data.groupby(['SUB_DISTRICT', 'REQUEST_TYPE']).size().reset_index(name='count')
    top_requests_data = top_requests_data.sort_values(by='count', ascending=False).groupby('SUB_DISTRICT').head(10)
    fig = px.bar(top_requests_data, x='REQUEST_TYPE', y='count', color='SUB_DISTRICT', barmode='group',
                 title="Top 10 Most Common Request Types by Sub-District")
    fig.update_layout(margin=dict(l=20, r=20, t=40, b=20))
    return fig

@app.callback(
    Output("day-of-week-chart", "figure"),
    Input("dummy", "n_clicks")
)
def update_day_of_week_chart(n_clicks):
    data['day_of_week'] = pd.to_datetime(data['CREATION_DATE']).dt.day_name()
    day_data = data['day_of_week'].value_counts().reindex(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']).reset_index()
    day_data.columns = ['Day of the Week', 'Service Requests']
    return generate_chart(day_data, chart_type="bar", x='Day of the Week', y='Service Requests', title="Service Requests by Day of the Week")

@app.callback(
    Output("hour-day-heatmap", "figure"),
    Input("dummy", "n_clicks")
)
def update_hour_day_heatmap(n_clicks):
    # Ensure 'CREATION_DATE' is in datetime format
    data['CREATION_DATE'] = pd.to_datetime(data['CREATION_DATE'], errors='coerce')  # Ensure datetime format

    # Filter data for 2023 and 2024
    data_filtered = data[data['CREATION_DATE'].dt.year.isin([2023, 2024])].copy()  # Make a copy to avoid SettingWithCopyWarning

    # Drop rows where 'CREATION_DATE' is NaT (invalid date entries)
    data_filtered = data_filtered.dropna(subset=['CREATION_DATE'])

    # Extract hour and day of the week from 'CREATION_DATE'
    data_filtered['hour'] = data_filtered['CREATION_DATE'].dt.hour
    data_filtered['day_of_week'] = data_filtered['CREATION_DATE'].dt.day_name()

    # Group data by day of week and hour to count requests
    heatmap_data = data_filtered.groupby(['day_of_week', 'hour']).size().reset_index(name='count')

    # Ensure the days of the week are in the correct order (Monday to Sunday)
    days_of_week_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    heatmap_data['day_of_week'] = pd.Categorical(heatmap_data['day_of_week'], categories=days_of_week_order, ordered=True)
    heatmap_data = heatmap_data.sort_values(by=['day_of_week', 'hour'])

    # Generate the heatmap using Plotly
    fig = px.density_heatmap(
        heatmap_data, 
        x='hour', 
        y='day_of_week', 
        z='count', 
        color_continuous_scale="Viridis", 
        title="Service Requests Heatmap by Hour and Day of the Week (2023 & 2024)"
    )

    # Update layout for better display
    fig.update_layout(margin=dict(l=20, r=20, t=40, b=20))

    return fig


@app.callback(
    Output("service-requests-month-chart", "figure"),
    Input("dummy", "n_clicks")
)
def update_service_requests_month_chart(n_clicks):
    # Ensure that 'CREATION_DATE' is parsed as datetime and handle any invalid dates
    data['CREATION_DATE'] = pd.to_datetime(data['CREATION_DATE'], errors='coerce')  # Ensure datetime format

    # Filter the data for the years 2023 and 2024
    data_filtered = data[data['CREATION_DATE'].dt.year.isin([2023, 2024])].copy()  # Make a copy to avoid SettingWithCopyWarning

    # Drop rows where 'CREATION_DATE' is NaT (invalid date entries)
    data_filtered = data_filtered.dropna(subset=['CREATION_DATE'])

    # Create a 'month' column formatted as 'YYYY-MM'
    data_filtered['month'] = data_filtered['CREATION_DATE'].dt.to_period('M').astype(str)

    # Count the number of requests per month
    month_data = data_filtered['month'].value_counts().sort_index().reset_index()
    month_data.columns = ['Month', 'Service Requests']

    # Ensure that all months from 2023 and 2024 are represented in the final data
    all_months = pd.date_range('2023-01-01', '2024-12-31', freq='MS').strftime('%Y-%m')  # Generate all months in 2023 and 2024
    all_months_df = pd.DataFrame(all_months, columns=['Month'])
    month_data = pd.merge(all_months_df, month_data, on='Month', how='left').fillna(0)  # Fill missing months with 0 service requests

    # Generate the bar chart
    fig = px.bar(month_data, x='Month', y='Service Requests', title="Service Requests by Month (2023 & 2024)")

    # Update layout for better display
    fig.update_layout(margin=dict(l=20, r=20, t=40, b=20))

    return fig






# Run the app
if __name__ == "__main__":
    app.run_server(debug=True)
