import requests
from geopy.geocoders import Nominatim

def get_census_tract(latitude, longitude):
    """
    Fetch the census tract code for given latitude and longitude using the
    U.S. Census Bureau Geocoding API.
    """
    # U.S. Census Bureau Geocoding API URL
    url = (
        "https://geocoding.geo.census.gov/geocoder/geographies/coordinates"
        f"?x={longitude}&y={latitude}&benchmark=Public_AR_Current"
        "&vintage=Current_Current&format=json"
    )
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        # Extract the Census Tract GEOID
        tract_info = data['result']['geographies']['Census Tracts'][0]
        tract_code = tract_info['GEOID']
        return tract_code
    except (KeyError, IndexError):
        return "Census tract not found for the given coordinates."
    except requests.exceptions.RequestException as e:
        return f"Error with the API request: {e}"

# Example Usage
if __name__ == "__main__":
    # Example coordinates (Times Square, NYC)
    lat = 35.01
    lon = -89.99
    tract_code = get_census_tract(lat, lon)
    print(f"Census Tract Code: {tract_code}")
