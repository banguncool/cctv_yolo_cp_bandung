# ==================================================================================================
# IMPORT LIBRARIES
# ==================================================================================================
import mvsdk
from mvsdk import CameraException
import platform
import numpy as np
import time
import cv2 as cv
import tools as tl
from concurrent.futures import ThreadPoolExecutor, as_completed

if cv.cuda.getCudaEnabledDeviceCount() > 0:
  print("\n== CUDA Version ==")
else:
  print("\n== CPU Version ==")

# ==================================================================================================
# DEFINED
# ==================================================================================================
PROD_MODE = True
ROTATE_CAMERA = False   # rotate camera 90 degree counterclockwise
BLUR = 5  # 0, 3, 5, 15, 25, 35, 45, 55, 65, 75, 85, 95, 105, etc

BUFFER_CAMERA_TIMOUT = 500   # in milliseconds

EXPOSURE_TIME = 40 * 1000   # in microseconds
EXPOSURE_GAIN = 5
GAMMA = 70
CONTRAST = 150

# Template matching multithreading settings
MAX_WORKER_THREADS = 8  # Maximum number of threads for template matching (adjust based on CPU cores)

FONT_SIZE = 0.8
VIEW_SCALE = 0.6
THICKNESS_THIN = 1
THICKNESS = 2
THICKNESS_BOLD = 3

COLOR_BLUE = (255, 0, 0)
COLOR_RED = (0, 0, 255)
COLOR_GREEN = (0, 255, 0)
COLOR_YELLOW = (0, 255, 255)
COLOR_PINK = (224, 25, 211)
COLOR_BROWN = (156, 156, 156)

RESOLUTION_WIDTH = 1280
RESOLUTION_HEIGHT = 1024


# ==================================================================================================
# PARAMETERS
# ==================================================================================================
FOV = [[195, 118], [1096, 928]]    # field of view 

refTopLeft = cv.imread('ref/top_left.png', cv.IMREAD_GRAYSCALE)
refTopRight = cv.imread('ref/top_right.png', cv.IMREAD_GRAYSCALE)
refBottomLeft = cv.imread('ref/bottom_left.png', cv.IMREAD_GRAYSCALE)
refBottomRight = cv.imread('ref/bottom_right.png', cv.IMREAD_GRAYSCALE)

refLabelTopLeft = cv.imread('ref/label_top_left.png', cv.IMREAD_GRAYSCALE)
refLabelTopRight = cv.imread('ref/label_top_right.png', cv.IMREAD_GRAYSCALE)
refLabelBottomLeft = cv.imread('ref/label_bottom_left.png', cv.IMREAD_GRAYSCALE)
refLabelBottomRight = cv.imread('ref/label_bottom_right.png', cv.IMREAD_GRAYSCALE)

# ==================================================================================================
# VARIABLES
# ==================================================================================================
np.set_printoptions(suppress=True)

hCamera = 0
pFrameBuffer = 0
frame = tl.blankBlack(RESOLUTION_WIDTH, RESOLUTION_HEIGHT)
if ROTATE_CAMERA: frame = cv.rotate(frame, cv.ROTATE_90_COUNTERCLOCKWISE)
original = frame.copy()

textMousePosition = '(0, 0)'
meassurePoint = [[0, 0], [0, 0]]
meassureEvent = 0

if PROD_MODE == True:
  print('Production mode')
  print('Trigger mode: Hardware')
else:
  print('Development mode')
  print('Press "c" to capture image')

print('Press "ESC" to exit program')


# ==================================================================================================
# DEFINITION
# ==================================================================================================
# mouse callback features --------------------------------------------------------------------------
def mouseRectangle(event, x, y, flags, param):
  global meassurePoint
  global meassureEvent
  global VIEW_SCALE
  global textMousePosition
  
  scale = 1 / VIEW_SCALE
  if event == cv.EVENT_MOUSEMOVE:
    textMousePosition = f'({int(x*scale)}, {int(y*scale)})'
  else:
    textMousePosition = ''

  # point 1
  if meassureEvent == 0:
    if event == cv.EVENT_LBUTTONDOWN:
      meassurePoint[0][0] = int(x / VIEW_SCALE)
      meassurePoint[0][1] = int(y / VIEW_SCALE)
      meassurePoint[1][0] = int(x / VIEW_SCALE)
      meassurePoint[1][1] = int(y / VIEW_SCALE)

  # point 2
  if meassureEvent == 1:
    if event == cv.EVENT_MOUSEMOVE:
      meassurePoint[1][0] = int(x / VIEW_SCALE)
      meassurePoint[1][1] = int(y / VIEW_SCALE)

  if event == cv.EVENT_LBUTTONDOWN:
    if meassureEvent == 1:
      print(meassurePoint)

    meassureEvent += 1 
    if meassureEvent > 2:
      meassureEvent = 0
      meassurePoint = [[0, 0], [0, 0]]

# loop program -------------------------------------------------------------------------------------
def loop():
  global hCamera
  global frame, original

  # mouse position and rectangle
  global textMousePosition
  global meassurePoint
  global meassureEvent

  cv.namedWindow('Camera')

  while True:   # ESC key
    try:
      if PROD_MODE == False:
        # set mouse callback
        cv.setMouseCallback('Camera', mouseRectangle)

        frame = cv.putText(frame, textMousePosition, (20, 50),
          cv.FONT_HERSHEY_SIMPLEX, FONT_SIZE, (0, 255, 255), THICKNESS)
        
        if meassureEvent != 0:
          ms = meassurePoint
          cv.putText(frame, f'M: {meassureEvent}: ({ms[0][0]}, {ms[0][1]}) ({ms[1][0]}, {ms[1][1]})', 
              (20, 100), cv.FONT_HERSHEY_SIMPLEX, FONT_SIZE, COLOR_YELLOW, THICKNESS)
          cv.rectangle(frame, meassurePoint[0], meassurePoint[1], COLOR_GREEN, THICKNESS)

      # show image
      cv.imshow('Camera', tl.resize(frame, VIEW_SCALE))
      
      # keyboard event -----------------------------------------------------------------------------
      keyPress = cv.waitKey(1) & 0xFF
      if PROD_MODE == False:
        # waitKey in order to ensure that the image redraw
        if keyPress != 255: 
          print(f'Keypress: {keyPress}')

          # save image press "c"
          if keyPress == 67 or keyPress == 99:
            print('Capture image')
            tl.capture(original)
            frameCapture = frame.copy()
            if ROTATE_CAMERA:
              frameCapture = cv.putText(frameCapture, 'Capture', (RESOLUTION_HEIGHT - 150, 50),
                  cv.FONT_HERSHEY_SIMPLEX, FONT_SIZE, (0, 255, 255), THICKNESS)
            else:
              frameCapture = cv.putText(frameCapture, 'Capture', (50, 50),
                  cv.FONT_HERSHEY_SIMPLEX, FONT_SIZE, (0, 255, 255), THICKNESS)
            
            cv.imshow('Camera', tl.resize(frameCapture, VIEW_SCALE))
            cv.waitKey(1)   # give time opencv to redraw
            time.sleep(1)

            cv.imshow('Camera', tl.resize(frame, VIEW_SCALE))
            print('Done capture image')

      # esc to exit program
      if keyPress == 27:
        break;
          
      # get captured image -------------------------------------------------------------------------
      if PROD_MODE == False: mvsdk.CameraSoftTrigger(hCamera)
      pRawData, FrameHead = mvsdk.CameraGetImageBuffer(hCamera, BUFFER_CAMERA_TIMOUT)

      timeStart = time.time()
      mvsdk.CameraImageProcess(hCamera, pRawData, pFrameBuffer, FrameHead)
      mvsdk.CameraReleaseImageBuffer(hCamera, pRawData)

      if platform.system() == "Windows":
        mvsdk.CameraFlipFrameBuffer(pFrameBuffer, FrameHead, 1)

      frame_data = (mvsdk.c_ubyte * FrameHead.uBytes).from_address(pFrameBuffer)
      frame = np.frombuffer(frame_data, dtype=np.uint8)
      frame = frame.reshape((FrameHead.iHeight, FrameHead.iWidth,
          1 if FrameHead.uiMediaType == mvsdk.CAMERA_MEDIA_TYPE_MONO8 else 3))

      if ROTATE_CAMERA: frame = cv.rotate(frame, cv.ROTATE_90_COUNTERCLOCKWISE)

      if BLUR > 0: frame = cv.GaussianBlur(frame, (BLUR, BLUR), 0)
      original = frame.copy()

      print('Frame size:', frame.shape)

      # do algorithm job here ----------------------------------------------------------------------
      if PROD_MODE == True:
        # using multithreaded template matching
        gray = cv.cvtColor(original, cv.COLOR_BGR2GRAY)
        timeStart = tl.millis()

        # prepare all templates for multithreaded processing
        templates = {
          'top_left': refTopLeft,
          'top_right': refTopRight,
          'bottom_left': refBottomLeft,
          'bottom_right': refBottomRight,
          'label_top_left': refLabelTopLeft,
          'label_top_right': refLabelTopRight,
          'label_bottom_left': refLabelBottomLeft,
          'label_bottom_right': refLabelBottomRight
        }

        # perform all template matching operations in parallel
        results = tl.templateMatchingMultithreaded(gray, templates, confidence=80, max_workers=MAX_WORKER_THREADS)

        # extract results for outer corners
        conf_tl, rec_tl = results['top_left']
        conf_tr, rec_tr = results['top_right'] 
        conf_bl, rec_bl = results['bottom_left']
        conf_br, rec_br = results['bottom_right']

        # extract results for label corners
        conf_ltl, rec_ltl = results['label_top_left']
        conf_ltr, rec_ltr = results['label_top_right']
        conf_lbl, rec_lbl = results['label_bottom_left']
        conf_lbr, rec_lbr = results['label_bottom_right']

        # draw outer corners and calculate centers
        cv.rectangle(frame, rec_tl[0], rec_tl[1], COLOR_BROWN, THICKNESS_THIN)
        centerTopLeft = tl.getCenterPoint(rec_tl[0], rec_tl[1])
        tl.drawCrossHair(frame, centerTopLeft, COLOR_BLUE, 10, THICKNESS)

        cv.rectangle(frame, rec_tr[0], rec_tr[1], COLOR_BROWN, THICKNESS_THIN)
        centerTopRight = tl.getCenterPoint(rec_tr[0], rec_tr[1])
        tl.drawCrossHair(frame, centerTopRight, COLOR_BLUE, 10, THICKNESS)

        cv.line(frame, centerTopLeft, centerTopRight, COLOR_PINK, THICKNESS)

        cv.rectangle(frame, rec_bl[0], rec_bl[1], COLOR_BROWN, THICKNESS_THIN)
        centerBottomLeft = tl.getCenterPoint(rec_bl[0], rec_bl[1])
        tl.drawCrossHair(frame, centerBottomLeft, COLOR_BLUE, 10, THICKNESS)

        cv.rectangle(frame, rec_br[0], rec_br[1], COLOR_BROWN, THICKNESS_THIN)
        centerBottomRight = tl.getCenterPoint(rec_br[0], rec_br[1])
        tl.drawCrossHair(frame, centerBottomRight, COLOR_BLUE, 10, THICKNESS)

        cv.line(frame, centerBottomLeft, centerBottomRight, COLOR_PINK, THICKNESS)

        # center position and out angle
        centerOutTop = tl.getCenterPoint(centerTopLeft, centerTopRight)
        tl.drawCrossHair(frame, centerOutTop, COLOR_BLUE, 10, THICKNESS)
        centerOutBottom = tl.getCenterPoint(centerBottomLeft, centerBottomRight)
        tl.drawCrossHair(frame, centerOutBottom, COLOR_BLUE, 10, THICKNESS)

        cv.line(frame, centerOutTop, centerOutBottom, COLOR_RED, THICKNESS)

        outAngle = tl.getRotationAngle(centerOutTop, centerOutBottom) - 90
        print(f'Angle: {outAngle}')

        # draw label corners and calculate centers
        cv.rectangle(frame, rec_ltl[0], rec_ltl[1], COLOR_BROWN, THICKNESS_THIN)
        centerLabelTopLeft = tl.getCenterPoint(rec_ltl[0], rec_ltl[1])
        tl.drawCrossHair(frame, centerLabelTopLeft, COLOR_BLUE, 10, THICKNESS)

        cv.rectangle(frame, rec_ltr[0], rec_ltr[1], COLOR_BROWN, THICKNESS_THIN)
        centerLabelTopRight = tl.getCenterPoint(rec_ltr[0], rec_ltr[1])
        tl.drawCrossHair(frame, centerLabelTopRight, COLOR_BLUE, 10, THICKNESS)

        cv.rectangle(frame, rec_lbl[0], rec_lbl[1], COLOR_BROWN, THICKNESS_THIN)
        centerLabelBottomLeft = tl.getCenterPoint(rec_lbl[0], rec_lbl[1])
        tl.drawCrossHair(frame, centerLabelBottomLeft, COLOR_BLUE, 10, THICKNESS)

        cv.rectangle(frame, rec_lbr[0], rec_lbr[1], COLOR_BROWN, THICKNESS_THIN)
        centerLabelBottomRight = tl.getCenterPoint(rec_lbr[0], rec_lbr[1])
        tl.drawCrossHair(frame, centerLabelBottomRight, COLOR_BLUE, 10, THICKNESS)

        cv.line(frame, centerLabelTopLeft, centerLabelBottomLeft, COLOR_PINK, THICKNESS)
        cv.line(frame, centerLabelTopRight, centerLabelBottomRight, COLOR_PINK, THICKNESS)

        centerLabelLeft = tl.getCenterPoint(centerLabelTopLeft, centerLabelBottomLeft)
        tl.drawCrossHair(frame, centerLabelLeft, COLOR_BLUE, 10, THICKNESS)
        centerLabelRight = tl.getCenterPoint(centerLabelTopRight, centerLabelBottomRight)
        tl.drawCrossHair(frame, centerLabelRight, COLOR_BLUE, 10, THICKNESS)

        cv.line(frame, centerLabelLeft, centerLabelRight, COLOR_RED, THICKNESS)
        pts = np.array([centerLabelTopLeft, centerLabelTopRight, centerLabelBottomRight, 
            centerLabelBottomLeft])
        cv.polylines(frame, [pts], True, COLOR_GREEN, THICKNESS)

        labelAngle = tl.getRotationAngle(centerLabelLeft, centerLabelRight)
        print(f'Angle: {labelAngle}')


        # draw line distance
        cv.line(frame, centerTopLeft, centerLabelTopLeft, COLOR_YELLOW, THICKNESS_THIN)
        cv.line(frame, centerTopRight, centerLabelTopRight, COLOR_YELLOW, THICKNESS_THIN)
        cv.line(frame, centerBottomLeft, centerLabelBottomLeft, COLOR_YELLOW, THICKNESS_THIN)
        cv.line(frame, centerBottomRight, centerLabelBottomRight, COLOR_YELLOW, THICKNESS)

        distanceTopLeft = tl.getDistance(centerTopLeft, centerLabelTopLeft)
        distanceTopRight = tl.getDistance(centerTopRight, centerLabelTopRight)
        distanceBottomLeft = tl.getDistance(centerBottomLeft, centerLabelBottomLeft)
        distanceBottomRight = tl.getDistance(centerBottomRight, centerLabelBottomRight)

        timeStop = tl.millis()

        # Calculate average confidence for display
        avg_conf = (conf_tl + conf_tr + conf_bl + conf_br + conf_ltl + conf_ltr + conf_lbl + conf_lbr) / 8
        print(f'Average Confidence: {avg_conf:.2f}')
        print(f'Time: {timeStop - timeStart} ms')

        frame = cv.putText(frame, f'Angle out: {round(outAngle, 2)}', (20, 50),
            cv.FONT_HERSHEY_SIMPLEX, FONT_SIZE, COLOR_YELLOW, THICKNESS)
        frame = cv.putText(frame, f'Angle label: {round(labelAngle, 2)}', (20, 100),
            cv.FONT_HERSHEY_SIMPLEX, FONT_SIZE, COLOR_YELLOW, THICKNESS)
        
        actualAngle = 0
        if outAngle > labelAngle:
          actualAngle = abs(labelAngle - outAngle) * -1
        else:
          actualAngle = abs(labelAngle - outAngle)


        frame = cv.putText(frame, f'Angle actual: {round(actualAngle, 2)}', (20, 150),
            cv.FONT_HERSHEY_SIMPLEX, FONT_SIZE, COLOR_YELLOW, THICKNESS)
        
        frame = cv.putText(frame, f'Dst 1: {round(distanceTopLeft, 1)}', (20, 200),
            cv.FONT_HERSHEY_SIMPLEX, FONT_SIZE, COLOR_YELLOW, THICKNESS)
        frame = cv.putText(frame, f'Dst 2: {round(distanceTopRight, 1)}', (20, 250),
            cv.FONT_HERSHEY_SIMPLEX, FONT_SIZE, COLOR_YELLOW, THICKNESS)
        frame = cv.putText(frame, f'Dst 3: {round(distanceBottomLeft, 1)}', (20, 300),
            cv.FONT_HERSHEY_SIMPLEX, FONT_SIZE, COLOR_YELLOW, THICKNESS)
        frame = cv.putText(frame, f'Dst 4: {round(distanceBottomRight, 1)}', (20, 350),
            cv.FONT_HERSHEY_SIMPLEX, FONT_SIZE, COLOR_YELLOW, THICKNESS)
        frame = cv.putText(frame, f'Time: {timeStop - timeStart} ms', (20, 400),
            cv.FONT_HERSHEY_SIMPLEX, FONT_SIZE, COLOR_YELLOW, THICKNESS)



        # using match ORB
        """
        print('start match orb')
        contourArea, transformedCorners, scale, angle = tl.matchHomography(refImg, original)
        
        print(f'Contour area: {contourArea}')

        if contourArea > 0:
          cv.polylines(frame, [np.int32(transformedCorners)], True, COLOR_GREEN, THICKNESS)
          frame = cv.putText(frame, f'Scale: {round(scale, 1)}', (50, 50),
              cv.FONT_HERSHEY_SIMPLEX, FONT_SIZE, COLOR_YELLOW, THICKNESS)
          frame = cv.putText(frame, f'Angle: {round(angle, 1)}', (50, 100),
              cv.FONT_HERSHEY_SIMPLEX, FONT_SIZE, COLOR_YELLOW, THICKNESS)
        """


    except CameraException as e:
      if e.error_code != -12: print(e.message)

    except Exception as e:
      print('Error:')
      print(e)
      time.sleep(1)


# set resolution -----------------------------------------------------------------------------------
def setResolution(width, height):
  res = mvsdk.CameraGetImageResolution(hCamera)
  x = int((res.iWidth - width) / 2)
  y = int((res.iHeight - height) / 2)

  mvsdk.CameraSetImageResolutionEx(
      hCamera, 0xFF, 0, 0, x, y, width, height, 0, 0)

# initializing -------------------------------------------------------------------------------------
def initializing():
  global RESOLUTION_WIDTH
  global RESOLUTION_HEIGHT

  try:
    print('Camera initializing')
    global hCamera
    global pFrameBuffer

    # shutdown the camera
    mvsdk.CameraUnInit(hCamera)

    # release buffer
    mvsdk.CameraAlignFree(pFrameBuffer)

    time.sleep(1)

    # get camera device
    deviceList = mvsdk.CameraEnumerateDevice()
    if deviceList == 0:
      print('No device found')
      return

    # turn on the camera
    hCamera = mvsdk.CameraInit(deviceList[0], -1, -1)

    # get camera description
    cap = mvsdk.CameraGetCapability(hCamera)

    # set weather this is mono camera or not
    monoCamera = (cap.sIspCapacity.bMonoSensor != 0)
    if monoCamera:
      mvsdk.CameraSetIspOutFormat(hCamera, mvsdk.CAMERA_MEDIA_TYPE_MONO8)
    else:
      mvsdk.CameraSetIspOutFormat(hCamera, mvsdk.CAMERA_MEDIA_TYPE_BGR8)

    # set camera in software trigger mode
    mvsdk.CameraSetTriggerMode(hCamera, 2)    # hardware trigger mode

    # set exposure
    mvsdk.CameraSetAeState(hCamera, 0)
    mvsdk.CameraSetExposureTime(hCamera, EXPOSURE_TIME)
    mvsdk.CameraSetAnalogGain(hCamera, EXPOSURE_GAIN)
    mvsdk.CameraSetGamma(hCamera, GAMMA)
    mvsdk.CameraSetContrast(hCamera, CONTRAST)
    mvsdk.CameraSetClrTempMode(hCamera, 1)    # set pure white color temperature

    # set camera resolution
    setResolution(RESOLUTION_WIDTH, RESOLUTION_HEIGHT)

    # camera start running thread from sdk
    mvsdk.CameraPlay(hCamera)

    # set buffer size for maximum camera resolution
    FrameBufferSize = cap.sResolutionRange.iWidthMax * \
        cap.sResolutionRange.iHeightMax * (1 if monoCamera else 3)

    # convert raw data to PC
    pFrameBuffer = mvsdk.CameraAlignMalloc(FrameBufferSize, 16)
    print('Done initializing')

    loop()
  except mvsdk.CameraException as e:
    print("CameraInit Failed({}): {}".format(e.error_code, e.message))
    time.sleep(3)
    initializing()


# ==================================================================================================
# PROGRAM START
# ==================================================================================================
initializing()