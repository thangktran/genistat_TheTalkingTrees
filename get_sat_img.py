from dotenv import load_dotenv
import googlemaps
import os
import io
import cv2
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import math

__DEBUG__ = False

load_dotenv() 
GOOGLE_MAP_API_KEY = os.getenv("GOOGLE_MAP_API_KEY")

if len(GOOGLE_MAP_API_KEY)==0:
    raise RuntimeError("GOOGLE_MAP_API_KEY is invalid")

gmaps = googlemaps.Client(key=GOOGLE_MAP_API_KEY)

HARD_CODE_CENTER = (320,320)
HARD_CODE_SIZE = (640, 640)

# return angle, area m^2, polygons, buildingMask, maskedImg
def get_sat_img(long, lat, zoom=19):

    responseRoadmap = gmaps.static_map(
            size=HARD_CODE_SIZE,
            zoom=zoom, # 5, 10, 15 street, 20 building
            center=(long,lat),
            maptype="roadmap", # roadmap, hybrid, satellite
            format="png",
            scale=1
        )
    responseSat = gmaps.static_map(
            size=HARD_CODE_SIZE,
            zoom=zoom, # 5, 10, 15 street, 20 building
            center=(long,lat),
            maptype="satellite", # roadmap, hybrid, satellite
            format="png",
            scale=1
        )

    roadMapInMemFile = _iterContentToInMemFile(responseRoadmap)
    satInMemFile = _iterContentToInMemFile(responseSat)

    roadMapImg = mpimg.imread(roadMapInMemFile)
    roadMapImg = _intervalMapping(roadMapImg, 0, 1, 0, 255)
    satImg = mpimg.imread(satInMemFile)
    satImg = _intervalMapping(satImg, 0, 1, 0, 255)

    allBuidingsMask = _getAllBuildingMask(roadMapImg)

    if __DEBUG__:
        plt.imshow(roadMapImg)
        plt.figure()
        plt.imshow(satImg)
        plt.figure()
        plt.imshow(allBuidingsMask)
        plt.figure()

    polygons, buildingMask = _getCenterPolygon(allBuidingsMask)
    maskedImg = cv2.bitwise_and(satImg, satImg, mask = buildingMask)

    area = 0
    angle = 0
    if polygons is not None:
        areaPixel = cv2.contourArea(polygons)
        # 140 pixel == 5m
        # 390 m^2 == 11502 pixel^2
        area = (areaPixel*390)/11502

        ###### CALCULATE ANGLE
        # longestEdge = np.array([0,0])
        # for i in range(1, len(polygons)):
        #     currentEdge = polygons[i][0]-polygons[i-1][0]
        #     if cv2.norm(currentEdge) > cv2.norm(longestEdge):
        #         longestEdge = currentEdge
        # ref = np.array([1,0])
        # angle = 180.0f/CV_PI * acos((reference.x*usedEdge.x + reference.y*usedEdge.y) / (cv::norm(reference) *cv::norm(usedEdge)));
        # angle = 180.0/3.14*math.acos(ref[0]*longestEdge[0] + ref[1]*longestEdge[1] / (cv2.norm(ref) * cv2.norm(longestEdge)))

    return angle, area, polygons, buildingMask, maskedImg


def _getCenterPolygon(mask):
    # apply thresholding to convert the grayscale image to a binary image
    ret,thresh = cv2.threshold(mask,50,255,0)
    # find the contours
    contours,hierarchy = cv2.findContours(thresh, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        approx = cv2.approxPolyDP(cnt, 0.01*cv2.arcLength(cnt, True), True)
        (x,y)=cnt[0,0]

        isInside = cv2.pointPolygonTest(approx, HARD_CODE_CENTER, False)

        if len(approx)>=4 and isInside>0:
            zerosMat = np.zeros(HARD_CODE_SIZE, dtype=np.uint8)
            buildingMask = cv2.drawContours(zerosMat, [approx], -1, 1, -1)
            return approx, buildingMask

    return None, None


def _intervalMapping(image, from_min, from_max, to_min, to_max):
    # map values from [from_min, from_max] to [to_min, to_max]
    # image: input array
    from_range = from_max - from_min
    to_range = to_max - to_min
    scaled = np.array((image - from_min) / float(from_range), dtype=float)
    result = to_min + (scaled * to_range)
    return np.array(result, dtype=int)


# return mask
def _getAllBuildingMask(img):
    l = np.array([240,240,240, 0])
    h = np.array([242,242,242, 255])
    mask = cv2.inRange(img, l, h)

    return mask


def _iterContentToInMemFile(iterContent):
    inMemoryFile = io.BytesIO()
    for chunk in iterContent:
        if chunk:
            inMemoryFile.write(chunk)

    inMemoryFile.seek(0)

    return inMemoryFile

def _getAngle(maskImg):
    pass

#### TEST ONLY
if __name__ == "__main__":

    __DEBUG__ = True

    angle, area, polygons, buildingMask, maskedImg = get_sat_img(48.354734, 11.798775, 16)

    print(f"{angle} : {area}")
    if area != 0:
        plt.imshow(buildingMask)
        plt.figure()
        plt.imshow(maskedImg)
    plt.show()

