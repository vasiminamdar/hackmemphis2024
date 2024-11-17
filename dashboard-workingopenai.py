import pandas as pd
import dash
from dash import dcc, html, Input, Output, State
import openai
from dash.exceptions import PreventUpdate
from dash import dash_table
from dash import html

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

# Extract latitude and longitude if the 'Location 1' column exists
if "Location 1" in open_requests.columns:
    open_requests[['longitude', 'latitude']] = open_requests['Location 1'].str.extract(r'POINT \(([^ ]+) ([^ ]+)\)')
    open_requests['latitude'] = pd.to_numeric(open_requests['latitude'], errors='coerce')
    open_requests['longitude'] = pd.to_numeric(open_requests['longitude'], errors='coerce')

# Initialize Dash app
app = dash.Dash(__name__)

# Set up the OpenAI API key (replace with your key)
openai.api_key = "sk-proj-"  # Replace with your OpenAI API key

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

# Pre-summarize datasets for quick use in chatbot
dataset_summaries = {
    "Open Requests": summarize_dataset(open_requests, "Open Requests"),
    "Closed Requests": summarize_dataset(closed_requests, "Closed Requests"),
    "Landbank Properties": summarize_dataset(landbank_properties, "Landbank Properties"),
    "Blight Neighborhood Dataset": summarize_dataset(blight_neighborhood_dataset, "Blight Neighborhood Dataset"),
    "Crime Dataset": summarize_dataset(crime_dataset, "Crime Dataset")
}

# App layout
app.layout = html.Div([
    html.H1("City Service Chat", style={"textAlign": "center", "fontSize": "32px"}),

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
])



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
            "Crime Dataset": crime_dataset,
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




# Run the app
if __name__ == "__main__":
    app.run_server(debug=True)
