from pyrosm import OSM
import pyrosm
# from shapely.geometry import Polygon, mapping
import shapely


def get_osm_data(dataset_name: str):
    osm_path = pyrosm.get_data("bremen")
    return OSM(osm_path)

def mapping(polygon):
    mapped = shapely.geometry.mapping(polygon)
    result = [(x[1], x[0]) for x in mapped["coordinates"][0]] if len(mapped["coordinates"]) == 1 else [(x[1], x[0]) for x in mapped["coordinates"]]
    # print(result)
    return result
    # {'type': 'FeatureCollection', 'features': [{'id': '164903', 'type': 'Feature', 'propeties': {}, 'geometry': {'type': 'Polygon', 
    # 'coordinates': (((8.8526926, 53.0928892), (8.852523, 53.092773), (8.8526579, 53.0927021), (8.852768, 53.0927775), (8.8528274, 53.0928182), (8.8526926, 53.0928892)),)}, 
    # 'bbox': (8.852523, 53.0927021, 8.8528274, 53.0928892)}], 'bbox': (8.852523, 53.0927021, 8.8528274, 53.0928892)}

    
# osm = OSM('/content/sample_data/test/bremen-latest.osm.pbf')