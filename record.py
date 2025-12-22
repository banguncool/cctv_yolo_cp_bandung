# ==================================================================================================
# CCTV Recording Program
# Created by: Ady Bangun (https://dolphinio.id)
# Date: 30 November 2025
#
# Description:
# Records CCTV footage from RTSP stream and saves to MP4 files
# Supports optional cropping based on config.ini settings
# ==================================================================================================

import cv2 as cv
import configparser
import ast
import os
from datetime import datetime
import time
import tools as tl

RECORD = False  # Start in view mode by default

# ==================================================================================================
# CONSTANTS
# ==================================================================================================
FONT_SIZE = 0.8
THICKNESS = 2
COLOR_YELLOW = (0, 255, 255)


# ==================================================================================================
# CONFIGURATION
# ==================================================================================================
# Read configuration
config = configparser.ConfigParser(inline_comment_prefixes=(';',))
config.read("config.ini")

# RTSP URL
rtspUrl = "rtsp://admin:adybangun12@192.168.0.64:554"

# Get crop settings from config
crop = config.getboolean("main", "crop")
cropArea = ast.literal_eval(config["main"]["cropArea"]) if crop else None
encode = config["main"]["encode"].lower()  # h264 / h265 / mjpeg
viewScale = float(config["ui"]["viewScale"])  # as a double value

# Video output settings
videoFolder = "video"
if not os.path.exists(videoFolder):
    os.makedirs(videoFolder)


# ==================================================================================================
# FUNCTIONS
# ==================================================================================================
def getFileName():
    """Generate filename in format YYMMDD-HHmm.mp4"""
    now = datetime.now()
    return now.strftime("%y%m%d-%H%M.mp4")


def getCurrentTime():
    """Get current time in HH:mm:ss format"""
    return datetime.now().strftime("%H:%M:%S")


# ==================================================================================================
# MAIN PROGRAM
# ==================================================================================================
def main():
    global RECORD
    isRecording = RECORD
    
    mode = "Recording" if isRecording else "Viewing"
    print(f"Starting CCTV {mode}...")
    print(f"RTSP URL: {rtspUrl}")
    print(f"Crop enabled: {crop}")
    if crop:
        print(f"Crop area: {cropArea}")
    
    # Open CCTV stream
    cap = cv.VideoCapture(rtspUrl)
    
    cap.set(cv.CAP_PROP_BUFFERSIZE, 3)
    
    if not cap.isOpened():
        print("Error: Cannot open RTSP stream")
        return
    
    # Get video properties
    fps = int(cap.get(cv.CAP_PROP_FPS))
    if fps == 0 or fps > 30:
        fps = 25  # Default FPS if not available
    
    width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    
    print(f"Stream properties: {width}x{height} @ {fps} FPS")
    
    # Adjust dimensions if cropping is enabled
    if crop:
        width = cropArea[1][0] - cropArea[0][0]
        height = cropArea[1][1] - cropArea[0][1]
        print(f"Output dimensions (cropped): {width}x{height}")
    
    # Initialize VideoWriter
    out = None
    outputPath = None
    
    def initializeRecorder():
        """Initialize video writer for recording"""
        nonlocal out, outputPath
        
        # Set video backend options - use more compatible codecs for Windows
        if encode == "h264":
            fourcc = cv.VideoWriter_fourcc(*'avc1')  # H264 - more compatible on Windows
        elif encode == "h265":
            fourcc = cv.VideoWriter_fourcc(*'mp4v')  # Fallback to mp4v for compatibility
            print("Warning: H265 not fully supported, using mp4v codec instead")
        elif encode == "mjpeg":
            fourcc = cv.VideoWriter_fourcc(*'MJPG')
        else:
            fourcc = cv.VideoWriter_fourcc(*'mp4v')
        
        # Create VideoWriter with fallback options
        fileName = getFileName()
        outputPath = os.path.join(videoFolder, fileName)
        
        # Try primary codec
        out = cv.VideoWriter(outputPath, fourcc, fps, (width, height))
        
        # If failed, try fallback codecs
        if not out.isOpened():
            print(f"Warning: Primary codec failed, trying mp4v fallback...")
            fourcc = cv.VideoWriter_fourcc(*'mp4v')
            out = cv.VideoWriter(outputPath, fourcc, fps, (width, height))
        
        if not out.isOpened():
            print(f"Warning: mp4v failed, trying XVID fallback...")
            fourcc = cv.VideoWriter_fourcc(*'XVID')
            outputPath = outputPath.replace('.mp4', '.avi')
            out = cv.VideoWriter(outputPath, fourcc, fps, (width, height))
        
        if not out.isOpened():
            print("Error: Cannot create video writer with any codec")
            return False
        
        print(f"Recording started: {outputPath}")
        return True
    
    # Initialize recorder if starting in record mode
    if isRecording:
        if not initializeRecorder():
            cap.release()
            return
    
    print("Press 'R' to start recording, 'V' for view mode, ESC to stop")
    
    windowName = "CCTV Recording" if RECORD else "CCTV View"
    cv.namedWindow(windowName)
    
    frameCount = 0
    startTime = time.time()
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            print("Error: Cannot read frame from stream")
            break
        
        # Skip corrupted frames
        if frame is None or frame.size == 0:
            continue
        
        frameCount += 1
        
        # Crop frame if enabled
        if crop:
            frameCropped = tl.crop(frame, cropArea[0], cropArea[1])
        else:
            frameCropped = frame
        
        # Ensure frame dimensions match VideoWriter
        if frameCropped.shape[1] != width or frameCropped.shape[0] != height:
            frameCropped = cv.resize(frameCropped, (width, height))
        
        # Write cropped frame to video file only if recording is active
        if isRecording and out is not None:
            out.write(frameCropped)
        
        # Create display frame (with running time overlay)
        displayFrame = frameCropped.copy()
        elapsed = time.time() - startTime
        hours = int(elapsed // 3600)
        minutes = int((elapsed % 3600) // 60)
        seconds = int(elapsed % 60)
        runningTime = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
        statusText = f"REC {runningTime}" if isRecording else f"VIEW {runningTime}"
        cv.putText(displayFrame, statusText, (10, 30), cv.FONT_HERSHEY_SIMPLEX, 
                   FONT_SIZE, COLOR_YELLOW, THICKNESS)
        
        # Show frame with time overlay
        cv.imshow(windowName, tl.resize(displayFrame, viewScale))
        
        # Display stats every 100 frames
        if frameCount % 100 == 0:
            elapsed = time.time() - startTime
            actualFps = frameCount / elapsed if elapsed > 0 else 0
            if isRecording:
                print(f"Recorded {frameCount} frames | FPS: {actualFps:.1f}")
            else:
                print(f"Viewed {frameCount} frames | FPS: {actualFps:.1f}")
        
        # Check for keyboard input
        key = cv.waitKey(1) & 0xFF
        
        if key == 27:  # ESC key
            mode = "Recording" if isRecording else "Viewing"
            print(f"\nStopping {mode.lower()}...")
            break
        elif key == ord('r') or key == ord('R'):  # R key - start recording
            if not isRecording:
                if initializeRecorder():
                    isRecording = True
                    startTime = time.time()  # Reset timer
                    frameCount = 0  # Reset frame count
                    print("Switched to RECORDING mode")
        elif key == ord('v') or key == ord('V'):  # V key - view mode
            if isRecording:
                if out is not None:
                    out.release()
                    print(f"Recording saved: {outputPath}")
                    out = None
                    outputPath = None
                isRecording = False
                startTime = time.time()  # Reset timer
                frameCount = 0  # Reset frame count
                print("Switched to VIEW mode")
    
    # Cleanup
    elapsed = time.time() - startTime
    mode = "Recording" if isRecording else "Viewing"
    print(f"\n{mode} finished:")
    print(f"Total frames: {frameCount}")
    print(f"Duration: {elapsed:.1f} seconds")
    print(f"Average FPS: {frameCount/elapsed:.1f}")
    if isRecording and outputPath:
        print(f"Saved to: {outputPath}")
    
    cap.release()
    if out is not None:
        out.release()
    cv.destroyAllWindows()


# ==================================================================================================
# ENTRY POINT
# ==================================================================================================
if __name__ == "__main__":
    main()
