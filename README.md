YOLOv8 Face & Motion Detection on NVIDIA Jetson Orin
This project provides a real-time face and motion detection solution optimized for NVIDIA Jetson Orin platforms. It leverages Ultralytics YOLOv8 for efficient object detection and combines it with OpenCV for motion detection and CUDA-accelerated image processing, all running on a TensorRT optimized engine.

✨ Key Features
YOLOv8 Face Detection: Real-time human face detection using a pre-trained YOLOv8 Nano face model (yolov8n-face.engine).
TensorRT Acceleration: Converts the ONNX model into a highly optimized TensorRT engine for maximum inference performance on Jetson Orin.
OpenCV Motion Detection: Implements background subtraction (MOG2) to identify moving regions in the video stream.
CUDA Accelerated Image Processing: Utilizes OpenCV's CUDA module for GPU-accelerated bilateral filtering, enhancing image quality.
Real-time FPS Display: Provides an on-screen display of the current frames per second (FPS).
USB Webcam Support: Designed to work seamlessly with standard USB cameras.
🛠️ Setup & Requirements
This project is developed and tested on an NVIDIA Jetson Orin platform running JetPack.

Hardware Requirements
NVIDIA Jetson Orin (Nano/NX/AGX) Developer Kit
USB Webcam
Software Requirements
Ensure your Jetson Orin has the following components installed and configured:

JetPack: This includes CUDA, cuDNN, and TensorRT.
Python 3.x: Python 3.8 or newer is recommended.
OpenCV: A version compiled with CUDA support (usually pre-installed with JetPack).
Ultralytics YOLOv8: Installed via pip.
Install Python Dependencies
Bash

# Install Ultralytics YOLOv8
pip install ultralytics
🚀 Model Download & Conversion
The official Ultralytics YOLOv8 models do not directly include a dedicated yolov8n-face.pt. This project utilizes an ONNX format model from the community, which then needs to be converted into a TensorRT engine for optimal performance on your Jetson Orin.

Download the ONNX Model:

Bash

wget https://github.com/akanametov/yolo-face/releases/download/v0.0.0/yolov8n-face.onnx
Convert to TensorRT Engine:
Use the trtexec tool, which is part of your JetPack installation, to compile the ONNX model into a TensorRT engine (.engine) file. This step is crucial for hardware-accelerated inference.

Bash

/usr/src/tensorrt/bin/trtexec --onnx=yolov8n-face.onnx --saveEngine=yolov8n-face.engine --fp16
--fp16: It's highly recommended to use FP16 precision for improved performance and lower memory usage on Jetson devices. If you encounter issues, try removing --fp16 to build with FP32 precision.
After executing this command, a file named yolov8n-face.engine will be generated in your current directory.
🏃 How to Run
Place the video_detection.py script in the same directory as your yolov8n-face.engine file.
Execute the script:
Bash

python3 video_detection.py
This will launch a new window displaying the real-time video feed from your USB webcam, overlaid with face detection bounding boxes, motion detection highlights, and the live FPS.

Press the q key to gracefully exit the application.
💡 Important Notes
Camera Permissions: Ensure your user account has the necessary permissions to access /dev/video0 (your USB webcam).
Performance: Actual FPS may vary depending on your specific Jetson Orin model, webcam resolution, lighting conditions, and the complexity of the scene (number of faces/amount of motion).
Error Handling: The to_gpu function includes basic fallback to CPU if CUDA operations fail, but a properly configured CUDA environment is essential for optimal performance.
Model Classes: The current video_detection.py explicitly checks for "face" and "hand" classes. If your specific yolov8n-face.engine provides different class names, you might need to adjust the if label in ["face", "hand"]: line accordingly.
🤝 Contributions
Feel free to open issues or submit pull requests if you have suggestions for improvements or bug fixes. Your contributions are welcome!
