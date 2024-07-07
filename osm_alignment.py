
import requests
import pandas as pd
import json
# import osmnx as ox
import xml.etree.ElementTree as ET

import osmpy
from shapely import wkt
from geopy.distance import geodesic
from math import pi, cos, sin
import pickle

def create_regular_polygon(lat, lon, side_length, num_sides):
    # Calculate the distance in meters for 1 degree of latitude and longitude
    lat_degree_distance = geodesic((lat, lon), (lat + 1, lon)).meters
    lon_degree_distance = geodesic((lat, lon), (lat, lon + 1)).meters

    # Calculate the angle between each vertex of the polygon
    angle = 2 * pi / num_sides

    # Calculate half of the side length in degrees
    half_side_length_degrees = side_length / 2.0 / lat_degree_distance

    # Calculate the coordinates of the polygon vertices
    vertices = []
    for i in range(num_sides):
        vertex_lat = lat + half_side_length_degrees * sin(i * angle)
        vertex_lon = lon + half_side_length_degrees * cos(i * angle)
        vertices.append((vertex_lat, vertex_lon))

    # Ensure the polygon is closed by making the last point the same as the first point
    vertices.append(vertices[0])

    return vertices


def create_polygon_corners(lat, lon, n_meters):
    # Calculate the distance in meters for 1 degree of latitude and longitude
    lat_degree_distance = geodesic((lat, lon), (lat + 1, lon)).meters
    lon_degree_distance = geodesic((lat, lon), (lat, lon + 1)).meters

    # Calculate half of the side length in degrees
    half_side_length_degrees = n_meters / 2.0 / lat_degree_distance

    # Calculate the coordinates of the four corners of the polygon
    top_left = (lat + half_side_length_degrees, lon - half_side_length_degrees)
    top_right = (lat + half_side_length_degrees, lon + half_side_length_degrees)
    bottom_left = (lat - half_side_length_degrees, lon - half_side_length_degrees)
    bottom_right = (lat - half_side_length_degrees, lon + half_side_length_degrees)

    # Ensure the polygon is closed by making the last point the same as the first point
    corners = [top_left, top_right, bottom_right, bottom_left, top_left]

    return corners




def get_osm_data(lat, lon, N_meters):
    bbox =create_polygon_corners(lat, lon, N_meters)
    # Format the bounding box coordinates as a WKT polygon string
    boundingbox = ', '.join([f'{lon} {lat}' for lat, lon in bbox])
    
    # Create the WKT polygon using the formatted string
    boundary = wkt.loads(f'POLYGON(({boundingbox}))')
    ammenities = osmpy.get('Amenities', boundary)
    # ammen_count = osmpy.get('AmentiesCount', boundary)
    # pois = osmpy.get('POIs', boundary)
    # roads = osmpy.get('RoadLength', boundary)
    return ammenities


def load_data_from_csv(csv_path, folder_path):
    # Read CSV
    df = pd.read_csv(csv_path)
    
    master_dict = {}
    
    for index, row in df.iterrows():
        # Construct full filepath
        pickle_path = folder_path + '/' + row['filepaths']
        
        # Load pickle file
        with open(pickle_path, 'rb') as file:
            pickle_data = pickle.load(file)
        
        # Extract keys from CSV and store data in the dictionary
        keys = [key.strip() for key in row['list_of_dict_keys'].split(',')]
        for key in keys:
            master_dict[key] = pickle_data.get(key, None)
    
    return master_dict
        