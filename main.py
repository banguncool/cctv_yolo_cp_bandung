# ==================================================================================================
# Created by: Ady Bangun (https://dolphinio.id)
# Date: 29 November 2025
#
# Requirements:
# - Ultralytics YOLOv11
# - Python 3.10
# - OpenCV 4.12.0
# - PyTorch with CUDA 12.6
# - configparser 7.2.0
# - deep-sort-realtime 1.3.2
#
# Version 1.0.0
# - Initial release
#
# ==================================================================================================

import platform
import numpy as np
import time
import cv2 as cv
import tools as tl
from ultralytics import YOLO
import random
import configparser
import ast


# ==================================================================================================
# DEFINED
# ==================================================================================================
FONT_SIZE = 0.8
THICKNESS_THIN = 1
THICKNESS = 2
THICKNESS_BOLD = 3

COLOR_BLUE = (255, 0, 0)
COLOR_RED = (0, 0, 255)
COLOR_GREEN = (0, 255, 0)
COLOR_YELLOW = (0, 255, 255)
COLOR_PINK = (224, 25, 211)
COLOR_BROWN = (156, 156, 156)


# ==================================================================================================
# VARIABLES
# ==================================================================================================
np.set_printoptions(suppress=True)
config = configparser.ConfigParser(inline_comment_prefixes=(';',))
config.read("config.ini")
# print(cfg["model"]["weights"])

resolutionWidth = 0
resolutionHeight = 0

model = YOLO(config["model"]["weights"])
cap = cv.VideoCapture(config["main"]["vidInput"])

# Get video resolution
resolutionWidth = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
resolutionHeight = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))

# Mouse callback variables
textMousePosition = '(0, 0)'
measurePoint = [[0, 0], [0, 0]]
measureEvent = 0

frameCount = 0
fpsStartTime = time.time()
fps = 0
fpsDisplay = 0
fpsFrameCount = 0
fpsUpdateTime = time.time()
fpsSet = int(config["ui"]["fpsSet"])
frameSkip = int(config["ui"]["frameSkip"])
viewScale = float(config["ui"]["viewScale"])
windowName = config["ui"]["windowName"]
drawRectangle = config.getboolean("main", "drawRectangle")
roi = ast.literal_eval(config["ui"]["roi"])
boundLine = ast.literal_eval(config["ui"]["boundLine"])
windowCropName = config["ui"]["windowCropName"]

boundLineRoi = [
  [
    roi[0][0] + boundLine[0],
    roi[0][1]
  ],
  [
    roi[0][0] + boundLine[1],
    roi[1][1]
  ]
]

if frameSkip > 0:
  fpsSet = fpsSet / (frameSkip + 1)


# ==================================================================================================
# DEFINITION
# ==================================================================================================
# Mouse callback function for rectangle drawing
def mouseRectangle(event, x, y, flags, param):
  global measurePoint
  global measureEvent
  global viewScale
  global textMousePosition
  
  scale = 1 / viewScale
  if event == cv.EVENT_MOUSEMOVE:
    textMousePosition = f'({int(x*scale)}, {int(y*scale)})'
  else:
    textMousePosition = ''

  # Point 1 - first click sets the first corner
  if measureEvent == 0:
    if event == cv.EVENT_LBUTTONDOWN:
      measurePoint[0][0] = int(x / viewScale)
      measurePoint[0][1] = int(y / viewScale)
      measurePoint[1][0] = int(x / viewScale)
      measurePoint[1][1] = int(y / viewScale)

  # Point 2 - mouse move updates the second corner
  if measureEvent == 1:
    if event == cv.EVENT_MOUSEMOVE:
      measurePoint[1][0] = int(x / viewScale)
      measurePoint[1][1] = int(y / viewScale)

  # Handle clicks
  if event == cv.EVENT_LBUTTONDOWN:
    if measureEvent == 1:
      # Print the two points to console
      print(measurePoint)

    measureEvent += 1 
    if measureEvent > 2:
      measureEvent = 0
      measurePoint = [[0, 0], [0, 0]]



# ==================================================================================================
# PROGRAM LOOP
# ==================================================================================================


# Calculate target frame time
targetFrameTime = 1.0 / fpsSet if fpsSet > 0 else 0
nextFrameTime = time.time()

# Set up mouse callback
cv.namedWindow(windowName)
cv.setMouseCallback(windowName, mouseRectangle)

while True:
  # Read frame
  ret, frame = cap.read()
  if not ret:
    break
  
  # Frame skipping
  frameCount += 1
  if frameSkip > 0 and frameCount % (frameSkip + 1) != 0:
    continue
  
  # Calculate actual FPS (measure displayed frames only)
  fpsEndTime = time.time()
  if fpsEndTime - fpsStartTime > 0:
    fps = 0.7 * fps + 0.3 * (1.0 / (fpsEndTime - fpsStartTime))  # Smoothed FPS
  fpsStartTime = fpsEndTime
  
  # Update FPS display every 1 second
  fpsFrameCount += 1
  if fpsEndTime - fpsUpdateTime >= 0.5:
    fpsDisplay = fpsFrameCount / (fpsEndTime - fpsUpdateTime)
    fpsFrameCount = 0
    fpsUpdateTime = fpsEndTime
  
  # Display FPS
  cv.putText(frame, f"FPS: {fpsDisplay:.1f}", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 
    FONT_SIZE, COLOR_YELLOW, THICKNESS)
  
  # Display mouse position
  if drawRectangle:
    cv.putText(frame, textMousePosition, (resolutionWidth - 150, 30), cv.FONT_HERSHEY_SIMPLEX, 
      FONT_SIZE, COLOR_YELLOW, THICKNESS)
    
    # Draw rectangle if measuring
    if measureEvent != 0:
      ms = measurePoint
      cv.putText(frame, f'M: {measureEvent}: ({ms[0][0]}, {ms[0][1]}) ({ms[1][0]}, {ms[1][1]})', 
          (resolutionWidth - 380, 70), cv.FONT_HERSHEY_SIMPLEX, FONT_SIZE, COLOR_YELLOW, THICKNESS)
      cv.rectangle(frame, measurePoint[0], measurePoint[1], COLOR_GREEN, THICKNESS)
  
  # ================================================================================================
  # PROCESSING
  # ================================================================================================
  crop = tl.crop(frame, roi[0], roi[1])

  cv.imshow(windowCropName, tl.resize(crop, viewScale))
  # draw rectangle around ROI



  cv.rectangle(frame, roi[0], roi[1], COLOR_BLUE, THICKNESS)
  # draw boundary line 
  cv.line(frame, (boundLineRoi[0][0], boundLineRoi[0][1]), (boundLineRoi[0][0], boundLineRoi[1][1]), COLOR_RED, THICKNESS_THIN)
  cv.line(frame, (boundLineRoi[1][0], boundLineRoi[0][1]), (boundLineRoi[1][0], boundLineRoi[1][1]), COLOR_RED, THICKNESS_THIN)


  
  # Show frame
  cv.imshow("CCTV YOLO Detection", tl.resize(frame, viewScale))
  # ================================================================================================
  # END OF PROCESSING
  # ================================================================================================

  # FPS limiting with accurate timing
  if fpsSet > 0:
    # Calculate time until next frame should be displayed
    currentTime = time.time()
    timeToWait = nextFrameTime - currentTime
    
    if timeToWait > 0:
      # Use busy-wait for last 2ms for better accuracy, sleep for the rest
      if timeToWait > 0.002:
        time.sleep(timeToWait - 0.002)
      
      # Busy-wait for remaining time
      while time.time() < nextFrameTime:
        pass
    
    # Set next frame time
    nextFrameTime += targetFrameTime
    
    # Reset if we're falling behind
    if nextFrameTime < currentTime:
      nextFrameTime = currentTime + targetFrameTime
    
    # Check for ESC key (use minimal wait)
    if cv.waitKey(1) & 0xFF == 27:
      break
  else:
    # ESC key to exit
    if cv.waitKey(1) & 0xFF == 27:
      break


# ==================================================================================================
# END
# ==================================================================================================
# Release resources
cap.release()
cv.destroyAllWindows()