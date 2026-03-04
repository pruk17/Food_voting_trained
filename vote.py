#python -m venv train_env
# Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
#.\train_env\Scripts\Activate.ps1
import tensorflow as tf
import pandas as pd
import cv2

print("TensorFlow version:", tf.__version__)
print("Pandas version:", pd.__version__)
print("OpenCV version:", cv2.__version__)
print("พร้อมเริ่มเทรนโมเดลแล้ว!")