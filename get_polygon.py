from pyrosm import OSM
import pandas as pd
import pickle
import os
from shapely.geometry import Polygon, mapping
import time
from kinh_utils import get_osm_data

# get_osm_final_building_list_coord_area(dataset_name)


def get_osm_df_building(dataset_name):
    data = get_osm_data(dataset_name)
    buildings = data.get_buildings()

    # buildings = buildings.drop(columns=['addr:country', 'addr:country', 'addr:full', 'addr:housenumber',
    #    'addr:housename', 'addr:postcode', 'addr:place', 'addr:street'], errors="ignore")

    buildings = buildings[['addr:city', 'addr:postcode', 'height', 'geometry']]
    return buildings


def create_dict(row):
    coord = mapping(row.loc["geometry"])
    area = row.geometry.area

    return {"coord": coord, "area": area}


def get_osm_final_building_list_coord_area(dataset_name):
    buildings = get_osm_df_building(dataset_name)
    print(buildings.shape)
    if not os.path.exists(f"pickle_buildings/{dataset_name}.pkl"):
        os.makedirs(f"pickle_buildings/", exist_ok=True)
        pickle.dump(buildings, open(f"pickle_buildings/{dataset_name}.pkl", "wb"))
    else:
        buildings = pickle.load(open(f"pickle_buildings/{dataset_name}.pkl", "rb"))
    
    buildings = buildings[['addr:city', 'addr:postcode', 'height', 'geometry']]

    results = buildings.apply(create_dict, axis=1).apply(pd.Series)

    os.makedirs(f"pickle_polygon/", exist_ok=True)
    pickle.dump(buildings, open(f"pickle_polygon/{dataset_name}.pkl", "wb"))

    return results


if __name__ == "__main__":
    start = time.time()
    get_osm_final_building_list_coord_area("bremen")   
    print(f"Elapsed time: {time.time() - start}")