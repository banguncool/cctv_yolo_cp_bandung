import cv2 as cv
import time
import numpy as np
from datetime import datetime
import os
import math
import random



# do nothing ---------------------------------------------------------------------------------------
def nothing(x):
  pass

# create blank black imagee ------------------------------------------------------------------------
def blankBlack(width, height):
  return np.zeros((height, width, 3), dtype=np.uint8)

# capture save in order to image to path -----------------------------------------------------------
def capture(frame):
  # check folder exists
  path = os.path.join(os.getcwd(), 'captured')
  if not os.path.exists(path):
    os.makedirs(path)

  today = datetime.today()
  today = today.strftime("%d%m%Y - %H%M%S")

  fileName = os.path.join(path, f'{today}.jpg')
  cv.imwrite(fileName, frame)

# resize -------------------------------------------------------------------------------------------
def resize(frame, scale):
  return cv.resize(frame, None, fx=scale, fy=scale, interpolation=cv.INTER_LINEAR)

# get milliseconds ---------------------------------------------------------------------------------
def millis():
  return round(time.time() * 1000)

# brightness and contrast --------------------------------------------------------------------------
def brightnessContrast(image, brightness=0, contrast=0):
  adjusted = cv.convertScaleAbs(image, alpha=contrast/127.0, beta=brightness)
  return adjusted

# convert color BGR to gray ------------------------------------------------------------------------
def grayAdjust(window, frame):
  blur = cv.getTrackbarPos('Blur', window)
  contast = cv.getTrackbarPos('Cont', window)
  brightness = cv.getTrackbarPos('Brig', window)

  frame = brightnessContrast(frame, brightness, contast)
  frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
  if blur != 0:
    frame = cv.GaussianBlur(frame, (5, 5), blur)

  return frame 

# threshold ----------------------------------------------------------------------------------------
def threshold(window, frame):
  thresh = cv.getTrackbarPos('Thres', window)
  _, frame = cv.threshold(frame, thresh, 255, cv.THRESH_BINARY_INV)
  return frame

# crop image ---------------------------------------------------------------------------------------
# ([x1:x2, y1:y2])
def crop(img, point1, point2):
  return img[point1[1]:point2[1], point1[0]:point2[0]]

# padding ------------------------------------------------------------------------------------------
def paddingCrop(img, top, right, bottom, left):
  height = img.shape[0]
  width = img.shape[1]

  x1 = top
  y1 = left
  x2 = height - bottom
  y2 = width - right

  frame = img[x1:x2, y1:y2]
  points = [[y1, x1], [y2, x2]]
  
  return frame, points


# template matching --------------------------------------------------------------------------------
# frame convert into gray = gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# template required cv.imread('template.jpg', cv.IMREAD_GRAYSCALE)
# confidence 0-100
def templateMatching(frame: cv.typing.MatLike, template, confidence = 50):
  confidence = confidence / 100

  fW, fH = template.shape[::-1]

  w, h = frame.shape[::-1]
  point1 = [0, 0]
  point2 = [w, h]

  conf = 0

  # for each in templates
  res = cv.matchTemplate(frame, template, cv.TM_CCOEFF_NORMED)
  loc = np.where(res >= confidence)
  
  for pt in zip(*loc[::-1]):
    if res[pt[1], pt[0]] > conf:
      conf = res[pt[1], pt[0]]
      point1 = [pt[0], pt[1]]
      point2 = [pt[0] + fW, pt[1] + fH]

  return conf * 100, [point1, point2]

# find difference ----------------------------------------------------------------------------------
def findDifference(image1, image2, threshold, minCountour, minWidth, maxWidth, minHeight, maxHeight):
  image1 = cv.cvtColor(image1, cv.COLOR_BGR2GRAY)
  image2 = cv.cvtColor(image2, cv.COLOR_BGR2GRAY)

  # Compute the absolute difference
  difference = cv.absdiff(image1, image2)

  # Apply a threshold to get a binary image
  _, thresh = cv.threshold(difference, threshold, 200, cv.THRESH_BINARY)

  # Find contours of the differences
  contours, _ = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

  result = []
  for cnt in contours:
    if len(cnt) > minCountour:
      get = False
      (x, y, w, h) = cv.boundingRect(cnt)
      if w > minWidth and w <= maxWidth:
        get = True 
      
      if h > minHeight and h <= maxHeight:
        get = True
        
      if get:
        result.append(cnt)

  return result

# move contour -------------------------------------------------------------------------------------
def moveContour(contour, x, y):
  for i in range(len(contour)):
    contour[i][0][0] += x  # Update x coordinate
    contour[i][0][1] += y  # Update y coordinate
  
  return contour

def borderResult(img, good = False):
  height = img.shape[0]
  width = img.shape[1]

  if good:
    img = cv.putText(img, 'GOOD', (20, 50),
        cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    return cv.rectangle(img, [0, 0], [width, height], (0, 255, 0), 14)
  else:
    img = cv.putText(img, 'NOT GOOD', (20, 50),
        cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    return cv.rectangle(img, [0, 0], [width, height], (0, 0, 255), 14)
  
# match detection with ORB in order to get orientation and scale -----------------------------------
def matchHomography(img1, img2, MIN_MATCH_COUNT = 10):

  img1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)   # queryImage
  img2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)   # trainImage
  
  # Initiate SIFT detector
  sift = cv.SIFT_create()
  
  # find the keypoints and descriptors with SIFT
  kp1, des1 = sift.detectAndCompute(img1,None)
  kp2, des2 = sift.detectAndCompute(img2,None)
  
  FLANN_INDEX_KDTREE = 1
  index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
  search_params = dict(checks = 50)
  
  flann = cv.FlannBasedMatcher(index_params, search_params)
  
  matches = flann.knnMatch(des1,des2,k=2)
  
  # store all the good matches as per Lowe's ratio test.
  good = []
  for m,n in matches:
    if m.distance < 0.3*n.distance:
      good.append(m)

  print(f'len good: {len(good)}')
  
  if len(good) > MIN_MATCH_COUNT:
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    # Find homography using RANSAC
    H, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)
    matchesMask = mask.ravel().tolist()

    h, w = img1.shape
    pts = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)

    # Convert the homography matrix to an affine transformation matrix
    affine_matrix = H[:2, :]  # Extract the affine part

    # Transform template corners using the affine transformation
    dst = cv.transform(pts, affine_matrix)

    # Ensure the result is a symmetric rectangle
    contour = np.array(dst, dtype=np.int32)
    contourArea = cv.contourArea(contour)

    # Extract scale and rotation from the affine matrix
    scale_x = np.sqrt(affine_matrix[0, 0] ** 2 + affine_matrix[1, 0] ** 2)
    scale_y = np.sqrt(affine_matrix[0, 1] ** 2 + affine_matrix[1, 1] ** 2)
    scale = (scale_x + scale_y) / 2  # Average scale factor

    # Calculate rotation angle in degrees
    theta = np.arctan2(affine_matrix[1, 0], affine_matrix[0, 0])  # Extract rotation angle
    rotation_angle = np.degrees(theta)

    # Ensure the rectangle remains symmetric by adjusting to 2D form
    rect = cv.minAreaRect(contour)
    box = cv.boxPoints(rect)
    box = np.int0(box)

    return contourArea, box, scale, rotation_angle
  else:
    print(f"Not enough matches are found - {len(good)}/{MIN_MATCH_COUNT}")
    return 0, None, None, None
  
# get center point rectangle -----------------------------------------------------------------------
def getCenterPoint(pt1 = [0, 0], pt2 = [0,0 ]):
  x = (pt1[0] + pt2[0]) / 2
  y = (pt1[1] + pt2[1]) / 2

  return [int(x), int(y)]

# draw cross hair ----------------------------------------------------------------------------------
def drawCrossHair(img, centerPoint, color = (255, 0, 0), size = 5, thickness = 1):
  cp = centerPoint
  img = cv.line(img, (int(cp[0] - size), int(cp[1])), (int(cp[0] + size), int(cp[1])), color, thickness)
  img = cv.line(img, (int(cp[0]), int(cp[1] - size)), (int(cp[0]), int(cp[1] + size)), color, thickness)                             
  return img

# get angle of two points --------------------------------------------------------------------------
def getRotationAngle(pt1, pt2):
  return math.degrees(math.atan2(pt2[1] - pt1[1], pt2[0] - pt1[0]))

# get distance between two points ------------------------------------------------------------------
def getDistance(pt1, pt2):
  return math.sqrt((pt2[0] - pt1[0])**2 + (pt2[1] - pt1[1])**2)

# get visible random color -------------------------------------------------------------------------
def randomColor():
  r = random.randint(150, 200)
  g = random.randint(150, 200)
  b = random.randint(150, 200)
  return (b, g, r)