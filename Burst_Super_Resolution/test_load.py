import cv2
import os
from PIL import Image
import numpy as np

path = r'c:\Users\prana\Desktop\image\bipnet\Zurich-RAW-to-DSLR-Dataset\original_images\canon\107.jpg'

print(f"Testing path: {path}")
print(f"File exists: {os.path.exists(path)}")
print(f"File size: {os.path.getsize(path)} bytes")

try:
    print("Testing with OpenCV...")
    img_cv = cv2.imread(path)
    if img_cv is not None:
        print(f"OpenCV loaded successfully. Shape: {img_cv.shape}")
    else:
        print("OpenCV returned None")
except Exception as e:
    print(f"OpenCV failed: {e}")

try:
    print("Testing with Pillow...")
    img_pil = Image.open(path)
    img_pil_meta = np.array(img_pil)
    print(f"Pillow loaded successfully. Shape: {img_pil_meta.shape}")
except Exception as e:
    print(f"Pillow failed: {e}")
