import cv2
import numpy as np
from ultralytics import YOLO
import time
import sys

# Initialize YOLOv8 (TensorRT)
model = YOLO("yolov8n-face.engine") # Replace with yolov8n-hand.engine for hand detection

# Initialize Motion Detection (MOG2)
fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=100, detectShadows=False)

# Open USB Webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam (/dev/video0).")
    sys.exit(1)

# Set resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# CUDA Acceleration - Helper function to upload to GPU
def to_gpu(frame):
    try:
        gpu_frame = cv2.cuda_GpuMat()
        gpu_frame.upload(frame)
        return gpu_frame
    except cv2.error as e:
        print(f"CUDA Error: {e}. Falling back to CPU.")
        return frame

# Motion detection threshold
motion_threshold = 10000

# FPS calculation variables
prev_time = time.time()
frame_count = 0
fps = 0.0

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame.")
        break

    # Pre-process frame (CPU operation)
    frame = cv2.convertScaleAbs(frame, alpha=1.2, beta=10)

    # YOLOv8 Detection (uses GPU via TensorRT engine)
    results = model(frame, conf=0.5, iou=0.5)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = box.conf.item()
            cls = int(box.cls.item())
            label = model.names[cls]
            if label in ["face", "hand"]: # Assuming 'face' and 'hand' are desired detection classes
                color = (255, 0, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    # Motion Detection (CPU operation by default unless fgbg also supports CUDA)
    fgmask = fgbg.apply(frame)
    # Corrected: CHAIN_APPROX_SIMPLE spelling
    contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        if cv2.contourArea(contour) > motion_threshold:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(frame, "Motion", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    # CUDA Processing for image filtering
    # Upload frame to GPU
    gpu_frame = to_gpu(frame)
    if isinstance(gpu_frame, cv2.cuda_GpuMat):
        # Create an output GpuMat for the result of the filter
        gpu_frame_filtered = cv2.cuda_GpuMat()
        # Apply the bilateral filter using explicit arguments
        cv2.cuda.bilateralFilter(src=gpu_frame, dst=gpu_frame_filtered, d=5, sigmaColor=50, sigmaSpace=50)
        # Update gpu_frame to point to the filtered result
        gpu_frame = gpu_frame_filtered
        # Download the processed frame back to CPU for display
        frame = gpu_frame.download()
    else:
        # If CUDA fails, ensure 'frame' variable is still the CPU frame
        pass # frame already holds the CPU frame in this case

    # Calculate and Display FPS (CPU operation)
    frame_count += 1
    curr_time = time.time()
    if curr_time - prev_time >= 1.0:
        fps = frame_count / (curr_time - prev_time)
        frame_count = 0
        prev_time = curr_time

    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    # Display frame (CPU operation)
    cv2.imshow("Video Detection", frame)

    # Exit on 'q'
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
