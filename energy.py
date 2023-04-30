import numpy as np
import math
from pyproj import Transformer
import random
import pandas as pd
import time

# Define coordinate systems
from_crs = "EPSG:4326"  # WGS 84
to_crs = "EPSG:31467"  # Gauss Krüger Zone 3
# Information extracted from the dataset header
XLLCORNER = 3280500
YLLCORNER = 5237500
NROWS = 866
CELLSIZE = 1000
NODATA_VALUE = -999
data = np.loadtxt("grids_germany_annual_radiation_global_2022.asc", skiprows=28)

def get_sun_radiance(row):
    lat, long = row["lat"], row["long"]
    # Coordinates of TU Munich
    # latitude, longitude = 48.1496636, 11.5656715

    # Create transformer object
    transformer = Transformer.from_crs(from_crs, to_crs)

    # Convert latitude and longitude to Gauss Krüger coordinates
    h, r = transformer.transform(lat, long)

    data[data == -999] = np.nan

    y, x = math.floor((r - XLLCORNER) / CELLSIZE), NROWS - math.ceil((h - YLLCORNER) / CELLSIZE)
    radiance = data[x, y]  # kWh/m^2/year
    return radiance


def _get_potential_energy(row, area, lat, long, efficient_factor, rooftype_factor, azimuth, tilt_angle):
#     radiance = get_sun_radiance(row)
    radiance = row['sun_radian']

    optimal_angle_factor = math.cos(math.radians(azimuth)) * math.cos(math.radians(tilt_angle))

    total_energy = rooftype_factor * efficient_factor * radiance * area * optimal_angle_factor 
    return total_energy  # kWh/year

def get_potential_energy(row):
    # optimal_angle_factor = math.cos(math.radians(azimuth)) * math.cos(math.radians(tilt_angle))
    rooftype_factor = 0.8 if row["type"] == "flat" else 0.4

    total_energy = _get_potential_energy(row, row["area"], row["lat"], row["long"], efficient_factor=0.2, rooftype_factor=rooftype_factor, azimuth=0, tilt_angle=30)
    return total_energy

def _get_efficacy(buildings, total_solar_area, efficient_factor=0.2, azimuth=0, tilt_angle=30):
    expected_total = 0
    for building in buildings:
        expected_total += _get_potential_energy(data, building.area, building.lat, building.long, efficient_factor, azimuth, tilt_angle)

    return expected_total / total_solar_area  # kWh/m^2/year

def main():
    df = pd.read_csv("germany.csv")
    df_city = df.groupby("city").agg({'long': 'mean', 'lat': 'mean'}).reset_index()
    df_city['sun_radian'] = df_city.apply(get_sun_radiance, axis=1)
    df = df.merge(df_city[['city', 'sun_radian']], on='city')
    df["Electric potential"] = df.apply(get_potential_energy, axis=1)
    df["solar_area"] = df.apply(lambda x: x["area"] * 0.8 if x["type"] == "flat" else 0.4 * x["area"], axis=1)
    df["efficiency"] = df["Electric potential"] / df["solar_area"]