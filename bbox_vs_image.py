import math
import cv2
import numpy as np


# Convert latitude and longitude to world coordinates
def lat_lng_to_world_coordinates(lat, lng, zoom):
    num_tiles = 2 ** zoom
    x = num_tiles * (lng + 180) / 360
    y = num_tiles * (1 - math.log(math.tan(lat * math.pi/180) + 1/math.cos(lat * math.pi/180)) / math.pi) / 2
    return x, y

# Convert world coordinates to pixel coordinates
def world_to_pixel_coordinates(world_x, world_y, zoom):
    pixel_x = world_x * 256
    pixel_y = world_y * 256
    return int(pixel_x), int(pixel_y)

# Convert latitude and longitude to pixel coordinates in the image
def lat_lng_to_pixel_coordinates(lat, lng, center_lat, center_lng, zoom, image_width, image_height):
    # Convert latitude and longitude of the center and the input point to world coordinates
    world_x_center, world_y_center = lat_lng_to_world_coordinates(center_lat, center_lng, zoom)
    world_x, world_y = lat_lng_to_world_coordinates(lat, lng, zoom)
    print("world_x:", world_x, "world_y:", world_y, "world_x_center:", world_x_center, "world_y_center:", world_y_center)
    
    # Difference between the world coordinates of the center and the input point
    world_x_diff = world_x - world_x_center
    world_y_diff = world_y - world_y_center

    # Convert world coordinates of the center and the input point to pixel coordinates
    pixel_x_center, pixel_y_center = world_to_pixel_coordinates(world_x_center, world_y_center, zoom)
    pixel_x, pixel_y = world_to_pixel_coordinates(world_x, world_y, zoom)

    # Calculate the pixel differences between the input point and the center point
    x_pixel_diff = pixel_x - pixel_x_center
    y_pixel_diff = pixel_y - pixel_y_center

    # Add half of the image size to obtain the pixel coordinates within the image
    x_pixel = x_pixel_diff + (image_width + 1) // 2
    y_pixel = y_pixel_diff + (image_height + 1) // 2

    return int(x_pixel), int(y_pixel)

zoom = 18
# 'bbox': (8.852523, 53.0927021, 8.8528274, 53.0928892)
# bbox = (8.852523, 53.0927021, 8.8528274, 53.0928892)
center_lat = 53.092744 
center_lng = 8.852580
lat1 = 53.0927021 # Lower left corner
lng1 = 8.852523
lat2 = 53.0928892 # Upper right corner
lng2 = 8.8528274
lower_left = (lat1, lng1)
upper_right = (lat2, lng2)
lower_right = (lat1, lng2)
upper_left = (lat2, lng1)
image_width = 512
image_height = 512

x1_pixel, y1_pixel = lat_lng_to_pixel_coordinates(lat1, lng1, center_lat, center_lng, zoom, image_width, image_height)
print(f"Pixel coordinates: ({x1_pixel}, {y1_pixel})")
x2_pixel, y2_pixel = lat_lng_to_pixel_coordinates(lat2, lng2, center_lat, center_lng, zoom, image_width, image_height)
print(f"Pixel coordinates: ({x2_pixel}, {y2_pixel})")

# Upper left corner

# Read the image
from PIL import Image
# How to install PIL in pip (Python 3.7.3): pip install Pillow
image_file = "satellite_image.jpg"
image = Image.open(image_file)
# Crop this image to get the building

# Convert to numpy array
image = np.array(image)
print("Image shape:", image.shape)

# Draw bounding box on image
cv2.rectangle(image, (x1_pixel, y1_pixel), (x2_pixel, y2_pixel), (0, 255, 0), 2)
# Draw x1, y1 pixel and x2, y2 pixel
cv2.circle(image, (x1_pixel, y1_pixel), 5, (0, 0, 255), -1)
cv2.circle(image, (x2_pixel, y2_pixel), 5, (0, 0, 255), -1)
# Draw center pixel
cv2.circle(image, (image_width // 2, image_height // 2), 5, (255, 0, 0), -1)
cv2.imshow("Image", image)
cv2.waitKey(0)


# print("Cropping image...", x1_pixel, y1_pixel, x2_pixel, y2_pixel)
# building_image = image.crop((x1_pixel, y2_pixel, x2_pixel, y1_pixel))

# Save the building image
# building_image.save("building.jpg")
