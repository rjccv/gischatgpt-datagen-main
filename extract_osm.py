from geo_dataloader import *
from torch.utils.data import DataLoader
import requests
import json
from dataclasses import dataclass, field
from transformers import HfArgumentParser
import textwrap
from tqdm import tqdm  
import re 
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import shutil
from itertools import islice
import tempfile
import pandas as pd



@dataclass
class OSMGen():
    dataset_name: str = field(metadata={'help':'Data set name'})
    csv_file:   str =  field( default=None, metadata={'help': 'CSV of im2gps3k'})
    im2gps_dir: str =  field(default=None, metadata={'help': 'Directory to the images'})
    # pipelining: bool = field( default=False, metadata={'help': 'Will not save the json if True'})
    method: int = field(default=1, metadata={'help':'Choose which method to use'})
    outfile: str = field(default='./')
    partition: int = field(default=1)





"""Method 1 - Raw osm data"""

def fetch_osm_data_raw(lat, lon, radius):
    # Define the Overpass API endpoint
    overpass_url = "http://overpass-api.de/api/interpreter"

    # Construct the query: fetch all elements within the specified radius around the lat/lon
    overpass_query = f"""
    [out:json];
    (
      node(around:{radius},{lat},{lon});
      way(around:{radius},{lat},{lon});
      relation(around:{radius},{lat},{lon});
    );
    out body;
    """
    # Fetch data from the API
    response = requests.get(overpass_url, params={'data': overpass_query})
    osm_data = response.json()

    # Extract the elements from the osm_data dictionary
    elements = osm_data.get("elements", [])

    return elements

"""Method 2 - Unique IDs"""
def fetch_osm_data_unique_ids(lat, lon, radius):
    # Define the Overpass API endpoint
    overpass_url = "http://overpass-api.de/api/interpreter"

    # Construct the query: fetch all elements within the specified radius around the lat/lon
    overpass_query = f"""
    [out:json];
    (
      node(around:{radius},{lat},{lon});
      way(around:{radius},{lat},{lon});
      relation(around:{radius},{lat},{lon});
    );
    out body;
    """
    # Fetch data from the API
    response = requests.get(overpass_url, params={'data': overpass_query})
    osm_data = response.json()

    # Extract the elements from the osm_data dictionary
    elements = osm_data.get("elements", [])

    # Create a dictionary to store unique elements based on their IDs
    unique_elements = {}

    # Iterate through the elements and store them in the dictionary
    for element in elements:
        unique_elements[element["id"]] = element

    # Return the list of unique elements
    return list(unique_elements.values())

"""Method 3 - Tags with Name"""

def fetch_osm_data_with_name(lat, lon, radius):
    # Define the Overpass API endpoint
    overpass_url = "http://overpass-api.de/api/interpreter"

    # Construct the query: fetch all elements within the specified radius around the lat/lon
    overpass_query = f"""
    [out:json];
    (
      node(around:{radius},{lat},{lon});
      way(around:{radius},{lat},{lon});
      relation(around:{radius},{lat},{lon});
    );
    out body;
    """
    # Fetch data from the API
    response = requests.get(overpass_url, params={'data': overpass_query})
    raw_osm_data = response.json()

    # Filter elements that have a "name" tag
    elements_with_name_tag = [element for element in raw_osm_data.get("elements", []) if "tags" in element and "name" in element["tags"]]

    # Convert the filtered elements into a list of dictionaries
    result_list = [{"element": element} for element in elements_with_name_tag]

    return result_list



"""Method 4 -Elements with more than 3 tags"""

def fetch_osm_data_over_three_tags(lat, lon, radius):
    # Define the Overpass API endpoint
    overpass_url = "http://overpass-api.de/api/interpreter"

    # Construct the query: fetch all elements within the specified radius around the lat/lon
    overpass_query = f"""
    [out:json];
    (
      node(around:{radius},{lat},{lon});
      way(around:{radius},{lat},{lon});
      relation(around:{radius},{lat},{lon});
    );
    out body;
    """

    # Fetch data from the API
    response = requests.get(overpass_url, params={'data': overpass_query})
    osm_data = response.json() 
    elements_with_tags = [element for element in osm_data["elements"] if "tags" in element]

    # tags that have more than 3 key
    multiple_key_tags = [element['tags'] for element in elements_with_tags if len(element['tags']) >3]


    return multiple_key_tags


"""Method 5 -Token reduction method - Raw (removing type, lat, lon, id tokens)"""

def fetch_osm_data_only_tags(lat, lon, radius):
    # Define the Overpass API endpoint
    overpass_url = "http://overpass-api.de/api/interpreter"

    # Construct the query: fetch all elements within the specified radius around the lat/lon
    overpass_query = f"""
    [out:json];
    (
      node(around:{radius},{lat},{lon});
      way(around:{radius},{lat},{lon});
      relation(around:{radius},{lat},{lon});
    );
    out body;
    """

    # Fetch data from the API
    response = requests.get(overpass_url, params={'data': overpass_query})
    osm_data = response.json()

    # Extract only the "tags" from the elements with tags
    tags_only = [element.get("tags", {}) for element in osm_data["elements"] if "tags" in element]

    tags_json = json.dumps(tags_only)

  # Load back as JSON object
    tags_obj = json.loads(tags_json)

    return tags_obj



"""Method 6 - Token reduction method - Processed """

def fetch_osm_data_tags_and_name(lat, lon, radius):
    # Define the Overpass API endpoint
    overpass_url = "http://overpass-api.de/api/interpreter"

    # Construct the query: fetch all elements within the specified radius around the lat/lon
    overpass_query = f"""
    [out:json];
    (
      node(around:{radius},{lat},{lon});
      way(around:{radius},{lat},{lon});
      relation(around:{radius},{lat},{lon});
    );
    out body;
    """

    # Fetch data from the API
    response = requests.get(overpass_url, params={'data': overpass_query})
    osm_data = response.json()

    # Extract only the "tags" that have a "name" key
    tags_with_name = [element["tags"] for element in osm_data["elements"] if "tags" in element and "name" in element["tags"]]

    return tags_with_name

"""Method 7 -Usefull Data - Processed """

def fetch_osm_data_method7(lat, lon, radius):

    # Define the Overpass API endpoint
    overpass_url = "http://overpass-api.de/api/interpreter"

    # Construct the query: fetch all elements within the specified radius around the lat/lon
    overpass_query = f"""
    [out:json];
    (
      node(around:{radius},{lat},{lon});
      way(around:{radius},{lat},{lon});
      relation(around:{radius},{lat},{lon});
    );
    out body;
    """

    tries = 0
    while True:
        if tries == 5:
            return None
        # Fetch data from the API
        # response = requests.get(overpass_url, params={'data': overpass_query})
        with requests.get(overpass_url, params={'data': overpass_query}, stream=True) as r:
            if r.status_code == 200:
                try:
                    osm_data = r.json()
                    break
                except:
                    tries += 1
                    print("Response is not in json format. Try again")
                    print(r.text)

    # Extract only the "tags" that have a "name" key
    filtered_tags = []
    for element in osm_data["elements"]:
        if "tags" in element and "name" in element["tags"]:
            
            tags = element["tags"]

            keys_to_remove = [
                'local_ref', 'brand:wikidata', 'ref', 'website', 'source',
                'wikipedia', 'wikidata', 'facebook', 'fhrs:id', 'image',
                'network:wikidata', 'network:wikipedia', 'check_date',
                'website:stock', 'contact:website', 'contact:facebook',
                'operator:wikidata', 'network:website', 'contact:twitter',
                'contact:github', 'contact:email', 'contact:youtube',
                'contact:instagram', 'contact:tiktok', 'contact:linkedin',
                'mapillary', 'artist:wikidata', 'contact:foursquare',
                'url', 'app:apple', 'app:google', 'opening_hours:url',
                'symbol', 'email', 'qroti:url','operator:wikipedia','contact:pinterest',
                'operator:wikipedia','operator:short'

                ]
            
            for key in keys_to_remove:
                tags.pop(key, None)


            tags = {key: value for key, value in tags.items() if "wikidata" not in key}
            tags = {key: value for key, value in tags.items() if "wikipedia" not in key}
            tags = {key: value for key, value in tags.items() if "multipolygon" not in key}

             # Keep 'name:en', remove other 'name:' keys
            name_en = tags.get('name:en')
            tags = {key: value for key, value in tags.items() if not key.startswith('name:') or key == 'name:en'}
            if name_en is not None:
                tags['name:en'] = name_en
            #tags = {key: value for key, value in tags.items() if not key.startswith('name:')} #To remove tags like 'name:en'
            tags = {key: value for key, value in tags.items() if not key.startswith('ref:')}
            tags = {key: value for key, value in tags.items() if not key.startswith('source:')}
            tags = {key: value for key, value in tags.items() if not key.startswith('brand:')}
            tags = {key: value for key, value in tags.items() if not key.startswith('gnis')} #Geographic Names Information System
            tags = {key: value for key, value in tags.items() if not key.startswith('gtfs')}
            tags = {key: value for key, value in tags.items() if not key.startswith('check_date:')}#last modification
            tags = {key: value for key, value in tags.items() if not key.startswith('tiger')}#"Topologically Integrated Geographic Encoding and Referencing
            tags = {key: value for key, value in tags.items() if not key.startswith('naptan')}
            tags = {key: value for key, value in tags.items() if (key.startswith("short_name") and key.endswith("en")) or not key.startswith("short_name")}
            tags = {key: value for key, value in tags.items() if (key.startswith("alt_name") and key.endswith("en")) or not key.startswith("alt_name")}
            tags = {key: value for key, value in tags.items() if (key.startswith("official_name") and key.endswith("en")) or not key.startswith("official_name")}
            tags = {key: value for key, value in tags.items() if (key.startswith("old_name") and key.endswith("en")) or not key.startswith("old_name")}

            filtered_tags.append(tags)

    return filtered_tags



def fetch_osm_data(lat, lon, radius=1000, method=1, dataset_name='MP16'):
    if method == 1:
        return fetch_osm_data_raw(lat, lon, radius)
    elif method == 2:
        return fetch_osm_data_unique_ids(lat, lon, radius)
    elif method == 3:
        return fetch_osm_data_with_name(lat, lon, radius)
    elif method == 4:
        return fetch_osm_data_over_three_tags(lat, lon, radius)
    elif method == 5:
        return fetch_osm_data_only_tags(lat, lon, radius)
    elif method == 6:
        return fetch_osm_data_tags_and_name(lat, lon, radius)
    elif method == 7:
        return fetch_osm_data_method7(lat, lon, radius)
    else:
        return "Invalid method"


import json

def extract_osm(csv_file, data_dir, method, output_dir, dataset_name, partition):
    os.makedirs(output_dir, exist_ok=True)

    im2gps3k = im2gps3ktestDataset(csv_file=csv_file, data_dir=data_dir)
    dataloader = DataLoader(im2gps3k, batch_size=1, shuffle=False)

    output_file = f'{output_dir}/M{method}_{dataset_name}_osm_data_{partition}.json'

    batch_size = 100  # Number of samples to process before writing to the file
    batch_counter = 0  # Counter for tracking the number of samples in the current batch
    first_write = True  # Flag to track if it's the first time writing to the file

    temp_file = tempfile.NamedTemporaryFile(delete=False, mode='w', encoding='utf-8', newline='', suffix='.json')
    try:
        with ThreadPoolExecutor(max_workers=5) as executor, open(temp_file.name, 'w') as temp_file_obj:
            futures_map = {}  # Use a dictionary to map futures to their data
            temp_file_obj.write('[')  # Start the JSON array

            first_write = True  # Track if it's the first write after opening the file
            batch_counter = 0

            existing_data = []
            # Read existing data from the output file if it exists
            if os.path.exists(output_file):
                with open(output_file, 'r') as file:
                    try:
                        existing_data = json.load(file)
                    except json.JSONDecodeError:
                        existing_data = []

                for entry in existing_data:
                    json.dump(entry, temp_file_obj)
                    temp_file_obj.write(',')
            
            first_write = not existing_data

            #checks existing filenames
            existing_filenames = {entry['filename'] for entry in existing_data}

            for img_id, lat, lon in tqdm(dataloader, desc="Fetching OSM data in parallel"):
                img_id, ext = os.path.splitext(img_id[0])

                filename = img_id + ext

                # Skip if the data is already collected
                if filename in existing_filenames:
                    continue


                future = executor.submit(fetch_osm_data, lat.item(), lon.item(), method=method)
                futures_map[future] = (img_id, ext, lat.item(), lon.item())
                batch_counter += 1


                if batch_counter >= batch_size:
                    # Process the current batch
                    for future in as_completed(futures_map):
                        img_id, ext, lat, lon = futures_map[future]
                        osm_data = future.result()

                        # Write data to file with proper JSON formatting
                        if not first_write:
                            temp_file_obj.write(',')
                        json.dump({'filename': img_id + ext, 'gps': (lat, lon), 'osm': osm_data}, temp_file_obj)
                        first_write = False

                    # Reset for the next batch
                    futures_map.clear()
                    batch_counter = 0

            # Process and write any remaining data
            for future in as_completed(futures_map):
                img_id, ext, lat, lon = futures_map[future]
                osm_data = future.result()

                if not first_write:
                    temp_file_obj.write(',')
                json.dump({'filename': img_id + ext, 'gps': (lat, lon), 'osm': osm_data}, temp_file_obj)

            temp_file_obj.write(']')  # End the JSON array

    finally:
        # Close the temp file and rename it to the final output file
        shutil.move(temp_file.name, output_file)


def main():
    parser = HfArgumentParser((OSMGen))
    
    osm_gen, _ = parser.parse_args_into_dataclasses(
            return_remaining_strings=True)    # Optionally, show a warning on unknown arguments.
    extract_osm(osm_gen.csv_file, osm_gen.im2gps_dir, osm_gen.method, osm_gen.outfile, osm_gen.dataset_name, osm_gen.partition)
    
    
if __name__ == "__main__":
    main()