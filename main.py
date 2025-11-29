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
RESOLUTION_WIDTH = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
RESOLUTION_HEIGHT = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))

# ==================================================================================================
# DEFINITION
# ==================================================================================================



# ==================================================================================================
# PROGRAM LOOP
# ==================================================================================================
frameCount = 0
fpsStartTime = time.time()
fps = 0
fpsDisplay = 0
fpsFrameCount = 0
fpsUpdateTime = time.time()
fpsSet = int(config["main"]["fpsSet"])
frameSkip = int(config["main"]["frameSkip"])
viewScale = float(config["ui"]["viewScale"])

# Calculate target frame time
targetFrameTime = 1.0 / fpsSet if fpsSet > 0 else 0
nextFrameTime = time.time()

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
  
  # ================================================================================================
  # PROCESSING
  # ================================================================================================



  
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