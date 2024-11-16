import requests
import pandas as pd

def fetch_data(api_url, limit, output_file):
    try:
        # Construct the API URL with $limit
        url = f"{api_url}?$limit={limit}"
        
        # Fetch data from API
        response = requests.get(url)
        response.raise_for_status()  # Raise an error for bad responses
        
        # Parse JSON response
        data = response.json()
        
        # Convert JSON data to a DataFrame
        df = pd.DataFrame(data)
        
        # Save to a CSV file
        df.to_csv(output_file, index=False)
        print(f"Successfully fetched and saved {len(df)} records to {output_file}")
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data from API: {e}")
    except Exception as e:
        print(f"Error processing data: {e}")

# API URLs
open_requests_url = "https://data.memphistn.gov/resource/aiee-9zqu.json"
closed_requests_url = "https://data.memphistn.gov/resource/2244-gnrp.json"

# Output file paths
open_requests_csv = "open_requests.csv"
closed_requests_csv = "closed_requests.csv"

# Fetch and save open and closed requests
fetch_data(open_requests_url, limit=50000, output_file=open_requests_csv)
fetch_data(closed_requests_url, limit=50000, output_file=closed_requests_csv)
