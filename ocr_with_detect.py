import os
import cv2
import numpy as np
import multiprocessing
from google.cloud import vision
from pdf2image import convert_from_path
import insightface

# Google Cloud Vision Credentials
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'mini-api-test-a0538c7dd495.json'

# Initialize InsightFace model
face_model = insightface.app.FaceAnalysis(providers=['CUDAExecutionProvider'])
face_model.prepare(ctx_id=0)

def extract_text_from_image(image):
    """Runs OCR on an in-memory image using Google Cloud Vision API."""
    client = vision.ImageAnnotatorClient()

    # Convert PIL image to a NumPy array
    image_np = np.array(image)

    # Encode the image as PNG for OCR
    _, encoded_img = cv2.imencode('.png', image_np)
    content = encoded_img.tobytes()
    image = vision.Image(content=content)

    response = client.text_detection(image=image)
    if response.error.message:
        raise Exception(f"Google Vision API Error: {response.error.message}")

    texts = response.text_annotations
    return texts[0].description if texts else "", texts

def recognize_faces(first_page):
    """Detects a face in the first page and saves the cropped face without saving full page."""
    img = np.array(first_page)  # Convert PIL image to NumPy array (RGB)
    
    faces = face_model.get(img)
    if not faces:
        print("No face detected in the first page.")
        return

    for face in faces:
        bbox = face.bbox.astype(int)
        x1, y1, x2, y2 = bbox

        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2

        fixed_width, fixed_height = 172, 216
        x1 = max(0, center_x - fixed_width // 2) + 3
        y1 = max(0, center_y - fixed_height // 2) - 4
        x2 = min(img.shape[1], x1 + fixed_width)
        y2 = min(img.shape[0], y1 + fixed_height)

        face_crop = img[y1:y2, x1:x2]
        
        # Convert from RGB to BGR before saving
        face_crop_bgr = cv2.cvtColor(face_crop, cv2.COLOR_RGB2BGR)
        cv2.imwrite("face_output.jpg", face_crop_bgr)
        cv2.imwrite("face_output.png", face_crop_bgr)

def process_first_page(pdf_path):
    """Converts the first page to an image and detects a face."""
    images = convert_from_path(pdf_path, dpi=150, first_page=1, last_page=1)
    if images:
        recognize_faces(images[0])

def process_page(image, page_num):
    """Processes a single page and runs OCR."""
    full_text, _ = extract_text_from_image(image)
    return page_num, full_text

def process_pdf(pdf_path):
    """Runs face detection on the first page and OCR on specific pages in parallel."""
    images = convert_from_path(pdf_path, dpi=150)
    num_pages = len(images)

    # Start face detection on first page in a separate process
    first_page_process = multiprocessing.Process(target=process_first_page, args=(pdf_path,))
    first_page_process.start()

    # Select second-last and third-last pages
    selected_pages = [(images[i], i + 1) for i in range(num_pages - 3, num_pages - 1)]

    # Run OCR in parallel
    with multiprocessing.Pool(processes=2) as pool:
        results = pool.starmap(process_page, selected_pages)

    # Wait for the first-page processing to finish
    first_page_process.join()

    # Print OCR results
    for result in results:
        page_num, full_text = result
        print(f"\nOCR Result for Page {page_num}:\n{full_text}")

def main():
    pdf_path = "정부24 - 유치원 및 초중등학교 학교(유치원)생활기록부 증명 _ 문서출력.pdf"
    process_pdf(pdf_path)

if __name__ == '__main__':
    main()
