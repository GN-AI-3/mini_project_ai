import time
import urllib
import cv2
import math
import numpy as np
import matplotlib.pyplot as plt
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import os

# IMAGE_FILENAMES should include the name of your image file.
IMAGE_FILENAMES = ['business-person.png']

# Desired height and width for resizing
DESIRED_HEIGHT = 480
DESIRED_WIDTH = 480

# Text to add below the image
TEXT = "Stylized Image"

# Directory to save processed images
OUTPUT_DIR = 'processed_images'

# Create the output directory if it doesn't exist
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# Performs resizing and showing the image
def resize_and_show(image):
    h, w = image.shape[:2]
    if h < w:
        img = cv2.resize(image, (DESIRED_WIDTH, math.floor(h/(w/DESIRED_WIDTH))))
    else:
        img = cv2.resize(image, (math.floor(w/(h/DESIRED_HEIGHT)), DESIRED_HEIGHT))
    
    # Use Matplotlib to display the image locally
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))  # Convert BGR to RGB for correct color display
    plt.axis('off')  # Hide axes
    plt.show()

# Load images into a dictionary
images = {name: cv2.imread(name) for name in IMAGE_FILENAMES}

# Display each image after resizing
for name, image in images.items():
    print(f"Processing: {name}")
    resize_and_show(image)

# Set up MediaPipe FaceStylizer for local use
base_options = python.BaseOptions(model_asset_path=r'C:\Users\201-18\NIP\models\face_stylizer_oil_painting.task')

options = vision.FaceStylizerOptions(base_options=base_options)

# Create the FaceStylizer
stylizer = vision.FaceStylizer.create_from_options(options)

# Function to stylize the images, add text, and save them locally
def stylize_images_with_text(image_filenames, stylizer, text, output_dir):
    stylized_images_with_text = []
    
    for image_file_name in image_filenames:
        # Read the image as a MediaPipe image
        image = mp.Image.create_from_file(image_file_name)

        # Stylize the image
        stylized_image = stylizer.stylize(image)

        # Convert to RGB for further processing or saving
        rgb_stylized_image = cv2.cvtColor(stylized_image.numpy_view(), cv2.COLOR_BGR2RGB)

        # Add text to the image
        h, w, _ = rgb_stylized_image.shape
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        color = (255, 255, 255)  # White color text
        thickness = 2
        text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
        text_x = (w - text_size[0]) // 2  # Center the text horizontally
        text_y = h - 20  # Place text near the bottom of the image

        # Add the text at the bottom
        cv2.putText(rgb_stylized_image, text, (text_x, text_y), font, font_scale, color, thickness)

        # Save the image locally
        output_path = os.path.join(output_dir, f"stylized_{os.path.basename(image_file_name)}")
        cv2.imwrite(output_path, cv2.cvtColor(rgb_stylized_image, cv2.COLOR_RGB2BGR))  # Convert back to BGR for saving
        
        # Append the processed image to the list
        stylized_images_with_text.append(rgb_stylized_image)
    
    return stylized_images_with_text

# 시간 측정 시작
start_time = time.time()

# Call the stylize_images_with_text function to get the processed images with text
stylized_images_with_text = stylize_images_with_text(IMAGE_FILENAMES, stylizer, TEXT, OUTPUT_DIR)

# 시간 측정 끝
end_time = time.time()

# 소요 시간 출력
elapsed_time = end_time - start_time
print(f"이미지 처리에 걸린 시간: {elapsed_time:.2f} 초")

# Display the stylized images with text
for img in stylized_images_with_text:
    resize_and_show(img)
