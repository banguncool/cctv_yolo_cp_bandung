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
import configparser
import ast
from sort.tracker import SortTracker



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

tracker = SortTracker(max_age=5, min_hits=3, iou_threshold=0.3)

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
confidence = float(config["model"]["confidence"])
direction = config["main"]["direction"].lower()  # left or right

boundLineRoi = [
  [roi[0][0] + boundLine[0], roi[0][1]],
  [roi[0][0] + boundLine[1], roi[1][1]]
]

# Counting variables
objectCount = 0
trackedObjects = {}  # Store track_id: {"passed_first": bool, "passed_second": bool, "counted": bool, "center_x": int}

if frameSkip > 0:
  fpsSet = fpsSet / (frameSkip + 1)

# Calculate target frame time
targetFrameTime = 1.0 / fpsSet if fpsSet > 0 else 0
nextFrameTime = time.time()


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


# update csv file ----------------------------------------------------------------------------------

# ==================================================================================================
# START & PROGRAM LOOP
# ==================================================================================================
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
  
  # Display object count
  cv.putText(frame, f"Count: {objectCount}", (10, 70), cv.FONT_HERSHEY_SIMPLEX, 
    FONT_SIZE, COLOR_GREEN, THICKNESS_BOLD)
  
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
  cropFrame = tl.crop(frame, roi[0], roi[1])

  # Use model.predict instead of model.track
  results = model.predict(cropFrame, conf=confidence, save=False, classes=0, verbose=False)

  # Prepare detections for sort-track
  detections = []
  for box in results[0].boxes:
    x_min, y_min, x_max, y_max = box.xyxy[0]
    conf = box.conf[0]
    cls = int(box.cls[0])
    
    # Convert to integers
    x_min, y_min, x_max, y_max = int(x_min), int(y_min), int(x_max), int(y_max)
    
    # Format: [x_min, y_min, x_max, y_max, confidence, class_id]
    detections.append([x_min, y_min, x_max, y_max, float(conf), cls])
  
  # Update tracker with detections (SortTracker expects detections and frame)
  if len(detections) > 0:
    tracks = tracker.update(np.array(detections), cropFrame)
  else:
    tracks = np.array([])
  
  # Process tracked objects
  for track in tracks:
    x_min, y_min, x_max, y_max, track_id = int(track[0]), int(track[1]), int(track[2]), int(track[3]), int(track[4])
    
    # Calculate center point of bounding box
    center_x = (x_min + x_max) // 2
    center_y = (y_min + y_max) // 2
    
    # Get absolute x positions of boundary lines in crop coordinates
    first_line_x = boundLine[0]
    second_line_x = boundLine[1]
    
    # Initialize tracking data for new objects
    if track_id not in trackedObjects:
      # Determine initial state based on where object first appears
      before_first = False
      between_lines = False
      after_second = False
      
      if direction == "right":
        # For right direction: first_line is on left, second_line is on right
        before_first = center_x < first_line_x
        between_lines = first_line_x <= center_x < second_line_x
        after_second = center_x >= second_line_x
      else:  # direction == "left"
        # For left direction: second_line is on left, first_line is on right
        before_first = center_x > first_line_x
        between_lines = second_line_x < center_x <= first_line_x
        after_second = center_x <= second_line_x
      
      trackedObjects[track_id] = {
        "before_first": before_first,
        "passed_first": False,
        "passed_second": False,
        "counted": False,
        "center_x": center_x
      }
    
    # Only process if object started before first line
    if trackedObjects[track_id]["before_first"]:
      # Check boundary crossing based on direction
      if direction == "right":
        # Moving from left to right (first line < second line)
        if center_x >= first_line_x and not trackedObjects[track_id]["passed_first"]:
          trackedObjects[track_id]["passed_first"] = True
        
        if trackedObjects[track_id]["passed_first"] and center_x >= second_line_x and not trackedObjects[track_id]["counted"]:
          trackedObjects[track_id]["counted"] = True
          objectCount += 1
      
      else:  # direction == "left"
        # Moving from right to left (second line > first line)
        if center_x <= first_line_x and not trackedObjects[track_id]["passed_first"]:
          trackedObjects[track_id]["passed_first"] = True
        
        if trackedObjects[track_id]["passed_first"] and center_x <= second_line_x and not trackedObjects[track_id]["counted"]:
          trackedObjects[track_id]["counted"] = True
          objectCount += 1
    
    # Update center position
    trackedObjects[track_id]["center_x"] = center_x
    
    # Draw bounding box
    color = COLOR_PINK if trackedObjects[track_id]["counted"] else COLOR_GREEN
    cv.rectangle(cropFrame, (x_min, y_min), (x_max, y_max), color, THICKNESS)
    
    # Draw track ID and center point
    cv.putText(cropFrame, f"ID:{track_id}", (x_min, y_min - 10), cv.FONT_HERSHEY_SIMPLEX, 
               0.5, color, THICKNESS_THIN)
    cv.circle(cropFrame, (center_x, center_y), 3, COLOR_RED, -1)

  
  # draw rectangle around ROI
  cv.rectangle(frame, roi[0], roi[1], COLOR_BLUE, THICKNESS)

  # draw boundary line 
  cv.line(frame, (boundLineRoi[0][0], boundLineRoi[0][1]), (boundLineRoi[0][0], boundLineRoi[1][1]), COLOR_RED, THICKNESS_THIN)
  cv.line(frame, (boundLineRoi[1][0], boundLineRoi[0][1]), (boundLineRoi[1][0], boundLineRoi[1][1]), COLOR_RED, THICKNESS_THIN)

  # Show frame
  cv.imshow(windowName, tl.resize(frame, viewScale))
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