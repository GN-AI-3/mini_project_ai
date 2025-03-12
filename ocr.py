import os
import cv2
import numpy as np
import multiprocessing
from google.cloud import vision
from pdf2image import convert_from_path

# Google Cloud Vision Credentials
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'mini-api-test-a0538c7dd495.json'

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

def process_page(image, page_num):
    """Processes a single page and runs OCR."""
    full_text, _ = extract_text_from_image(image)
    return page_num, full_text

def process_pdf(pdf_path):
    """Converts specific PDF pages into images and runs OCR in parallel."""
    images = convert_from_path(pdf_path, dpi=150)
    num_pages = len(images)

    # Select second-last and third-last pages
    selected_pages = [(images[i], i + 1) for i in range(num_pages - 3, num_pages - 1)]

    # Run OCR in parallel
    with multiprocessing.Pool(processes=2) as pool:
        results = pool.starmap(process_page, selected_pages)

    # Print OCR results
    for result in results:
        page_num, full_text = result
        print(f"\nOCR Result for Page {page_num}:\n{full_text}")

def main():
    pdf_path = "정부24 - 유치원 및 초중등학교 학교(유치원)생활기록부 증명 _ 문서출력.pdf"
    process_pdf(pdf_path)

if __name__ == '__main__':
    main()
