import requests
import pandas as pd

def download_file_from_drive(drive_link, output_path):
    """
    Download a file from a Google Drive link using requests.

    Args:
        drive_link (str): The Google Drive shareable link.
        output_path (str): The path where the downloaded file will be saved.
    """
    try:
        # Extract the file ID from the Google Drive link
        file_id = drive_link.split('/d/')[1].split('/view')[0]
        url = f"https://drive.google.com/uc?export=download&id={file_id}"
        
        # Send GET request to the file URL
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        # Save the file locally
        with open(output_path, "wb") as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)
        print(f"File downloaded successfully and saved to {output_path}")
    except Exception as e:
        print(f"Error downloading file: {e}")

def fetch_data(api_url, limit, output_file):
    """
    Fetch a large number of records from an API and save to a CSV file.

    Args:
        api_url (str): The API endpoint URL.
        limit (int): The number of records to fetch.
        output_file (str): The path to save the output CSV file.
    """
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

# File download details
landbank_properties_2024 = "https://drive.google.com/file/d/1UZGWpFTqS6AZ5a31DfGeTy_-j3MGsznu/view?usp=drive_link"
blight_neighborhood_dataset_2024 = "https://drive.google.com/file/d/1SykL7PDg7sJg34Jia-HwojiAK46oYL0f/view?usp=drive_link"

# File paths for local saving
landbank_file_path = "landbank_properties_2024.csv"
blight_neighborhood_file_path = "blight_neighborhood_dataset_2024.csv"

# Download files from Google Drive
download_file_from_drive(landbank_properties_2024, landbank_file_path)
download_file_from_drive(blight_neighborhood_dataset_2024, blight_neighborhood_file_path)

# API URLs
open_requests_url = "https://data.memphistn.gov/resource/aiee-9zqu.json"
closed_requests_url = "https://data.memphistn.gov/resource/2244-gnrp.json"

# Output file paths
open_requests_csv = "open_requests.csv"
closed_requests_csv = "closed_requests.csv"

# Fetch and save open and closed requests
fetch_data(open_requests_url, limit=50000, output_file=open_requests_csv)
fetch_data(closed_requests_url, limit=50000, output_file=closed_requests_csv)
