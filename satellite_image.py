from datetime import datetime

import requests

def get_satellite_image(api_key, latitude, longitude, zoom, width, height, format="jpg", save_as="satellite_image.jpg"):
    # Marker at the center of the image
    # url = f"https://maps.googleapis.com/maps/api/staticmap?center={latitude},{longitude}&zoom={zoom}&size={width}x{height}&markers=color:red%7C{latitude},{longitude}&key={api_key}&format={format}"
    # url = f"https://maps.googleapis.com/maps/api/staticmap?center={latitude},{longitude}&zoom={zoom}&size={width}x{height}&key={api_key}&format={format}"
    # url = f"https://maps.googleapis.com/maps/api/staticmap?center={latitude},{longitude}&zoom={zoom}&size={width}x{height}&maptype=hybrid&markers=color:red%7C{latitude},{longitude}&key={api_key}&format={format}"
    url = f"https://maps.googleapis.com/maps/api/staticmap?center={latitude},{longitude}&zoom={zoom}&size={width}x{height}&maptype=satellite&key={api_key}&format={format}"

    response = requests.get(url)
    
    if response.status_code == 200:
        with open(save_as, "wb") as f:
            f.write(response.content)
            print("Satellite image saved as satellite_image.jpg")
    else:
        print("Error fetching satellite image:", response.status_code)

api_key = "XXX"
lat1 = 53.0927021 # Lower left corner
lng1 = 8.852523
lat2 = 53.0928892 # Upper right corner
lng2 = 8.8528274
latitude = (lat1 + lat2) / 2
longitude = (lng1 + lng2) / 2
print("Center coordinates:", latitude, longitude)
zoom = 20 # For buildings
width = 1080
height = 1080

get_satellite_image(api_key, latitude, longitude, zoom, width, height)

