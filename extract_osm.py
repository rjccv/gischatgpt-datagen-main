from geo_dataloader import *
from torch.utils.data import DataLoader
import requests
import json
from dataclasses import dataclass, field
from transformers import HfArgumentParser
import textwrap
from tqdm import tqdm  
import re 


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
        response = requests.get(overpass_url, params={'data': overpass_query})
        if response.status_code == 200:
            try:
                osm_data = response.json()
                break
            except:
                tries += 1
                print("Response is not in json format. Try again")
                print(response.text)


    # Extract only the "tags" that have a "name" key
    filtered_tags = []
    for element in osm_data["elements"]:
        if "tags" in element and "name" in element["tags"]:
            
            tags = element["tags"]
            tags.pop("local_ref", None)  # Remove 'local_ref' if it exists
            tags.pop("brand:wikidata", None)      # Remove 'brand' if it exists
            tags.pop("ref", None)  
            tags.pop("website",None) 
            tags.pop("source",None)
            tags.pop("wikipedia",None)
            tags.pop("wikidata",None)
            tags.pop("facebook",None)
            tags.pop("fhrs:id",None)
            tags.pop("image",None)
            #tags.pop("contact:website",None)
            tags.pop("network:wikidata",None)
            tags.pop("network:wikipedia",None)
            tags.pop("fhrs:id",None)
            
            tags = {key: value for key, value in tags.items() if not key.startswith('name:')} #To remove tags like 'name:en'
            tags = {key: value for key, value in tags.items() if not key.startswith('ref:')}
            tags = {key: value for key, value in tags.items() if not key.startswith('source:')}
            tags = {key: value for key, value in tags.items() if not key.startswith('brand:')}
           
            
                 
            filtered_tags.append(tags)
    
    return filtered_tags

def fetch_osm_data(lat, lon, radius=1000, method=1, dataset_name='im2gps3k'):
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


def extract_osm(csv_file, data_dir, method, dataset_name, output_dir, partition):

    os.makedirs(output_dir, exist_ok=True)

    im2gps3k = im2gps3ktestDataset(csv_file=csv_file, data_dir=data_dir)
    # dataloader = DataLoader(im2gps3k, batch_size=1, shuffle=False) 
    dataloader = DataLoader(im2gps3k, batch_size=2, shuffle=False) 
   
   
   
    osm_data_list = []
    for img_id, lat, lon in tqdm(dataloader, desc="Appending OSM to dataset"):
            img_id , ext = os.path.splitext(img_id[0])

            # load language model for data enrichment
            osm_data = fetch_osm_data(lat.item(), lon.item(), method=method)
            if osm_data is None:
                continue
            osm_data_list.append({'filename': img_id + ext, 'gps':(lat.item(), lon.item()), 'osm':osm_data,})

    with open(f'{output_dir}/M{method}_{dataset_name}_osm_data_{partition}.json', 'w') as file:
        json.dump(osm_data_list, file)   


def main():
    parser = HfArgumentParser((OSMGen))
    
    osm_gen, _ = parser.parse_args_into_dataclasses(
            return_remaining_strings=True)    # Optionally, show a warning on unknown arguments.
    extract_osm(osm_gen.csv_file, osm_gen.im2gps_dir, osm_gen.method, osm_gen.dataset_name, osm_gen.outfile, osm_gen.partition)
    
    
if __name__ == "__main__":
    main()