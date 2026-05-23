import os
import json
from utilities.normalize_longitude import normalize_longitude


def process_coordinates(coords):
    if isinstance(coords, list) and len(coords) >= 2:
        if isinstance(coords[0], (int, float)) and isinstance(coords[1], (int, float)):
            # This is a coordinate pair [longitude, latitude, ...]
            normalized = [normalize_longitude(coords[0]), coords[1]]
            # Preserve additional dimensions (e.g., elevation)
            if len(coords) > 2:
                normalized.extend(coords[2:])
            return normalized
        else:
            # This is a nested list of coordinates
            return [process_coordinates(coord) for coord in coords]

    return coords


def process_geometry(geometry):
    if geometry['type'] == 'GeometryCollection':
        geometry['geometries'] = [process_geometry(g) for g in geometry['geometries']]
    else:
        geometry['coordinates'] = process_coordinates(geometry['coordinates'])

    return geometry


# Process files from both folders
folders = ["Data/raw/CHN/GeoJSON/", "Data/raw/SK/GeoJSON/"]

for folder in folders:
    for filename in os.listdir(folder):
        if filename.endswith(".geojson"):
            filepath = os.path.join(folder, filename)

            with open(filepath, 'r', encoding='utf-8') as f:
                geojson_data = json.load(f)

            # Handle Feature, FeatureCollection, or raw Geometry
            if geojson_data.get('type') == 'Feature':
                process_geometry(geojson_data['geometry'])
            elif geojson_data.get('type') == 'FeatureCollection':
                for feature in geojson_data['features']:
                    process_geometry(feature['geometry'])
            else:
                # Assume it's a raw Geometry object
                process_geometry(geojson_data)

            # Save with formatting
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(geojson_data, f, indent=2, ensure_ascii=False)
