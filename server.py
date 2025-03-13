import os
import uuid
import asyncio
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
from google.cloud import vision
from pdf2image import convert_from_bytes

import insightface
from fastapi import FastAPI, UploadFile, File

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'mini-api-test-a0538c7dd495.json'

vision_client = vision.ImageAnnotatorClient()

face_model = insightface.app.FaceAnalysis(providers=['CUDAExecutionProvider'])
face_model.prepare(ctx_id=0)

app = FastAPI()
executor = ThreadPoolExecutor(max_workers=4)

VERIFICATION_KEYWORDS = ["행동 특성 및 종합의견", "학교", "반"]

def verify_text(text):
    """Checks if extracted text contains all required keywords."""
    return all(keyword in text for keyword in VERIFICATION_KEYWORDS)

async def extract_text_from_image(image):
    """Runs OCR on an image using Google Cloud Vision API."""
    loop = asyncio.get_running_loop()
    image_np = np.array(image)
    return await loop.run_in_executor(executor, google_vision_ocr, image_np)

def google_vision_ocr(image_np):
    """Runs OCR on an image in a separate thread."""
    _, encoded_img = cv2.imencode('.png', image_np)
    content = encoded_img.tobytes()
    response = vision_client.text_detection(image=vision.Image(content=content))
    texts = response.text_annotations
    return texts[0].description if texts else ""

async def recognize_faces(image):
    """Detects a face in an image asynchronously."""
    loop = asyncio.get_running_loop()
    img = np.array(image)
    return await loop.run_in_executor(executor, detect_faces, img)

def detect_faces(img, save_dir="faces"):
    """Detects and crops face in an image and saves it with a unique filename."""
    faces = face_model.get(img)
    if not faces:
        return None

    face = faces[0]
    bbox = face.bbox.astype(int)
    x1, y1, x2, y2 = bbox

    center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
    fixed_width, fixed_height = 172, 216
    x1, y1 = max(0, center_x - fixed_width // 2) + 3, max(0, center_y - fixed_height // 2) - 4
    x2, y2 = min(img.shape[1], x1 + fixed_width), min(img.shape[0], y1 + fixed_height)

    face_crop = img[y1:y2, x1:x2]
    face_crop_bgr = cv2.cvtColor(face_crop, cv2.COLOR_RGB2BGR)

    # Ensure save directory exists
    os.makedirs(save_dir, exist_ok=True)

    # Generate a unique filename
    unique_id = uuid.uuid4().hex
    jpg_path = os.path.join(save_dir, f"face_{unique_id}.jpg")
    png_path = os.path.join(save_dir, f"face_{unique_id}.png")

    cv2.imwrite(jpg_path, face_crop_bgr)
    cv2.imwrite(png_path, face_crop_bgr)

    # Return file paths for further use
    return jpg_path, png_path

@app.post("/process-pdf/")
async def process_pdf(file: UploadFile = File(...)):
    """Processes a PDF, extracts face from first page, and OCR from last pages."""
    pdf_bytes = await file.read()
    images = convert_from_bytes(pdf_bytes, dpi=150)

    if not images:
        return {"error": "Failed to process PDF."}
    
    first_page = images[0]
    selected_pages = images[-3:-1] 
    
    face_task = asyncio.create_task(recognize_faces(first_page))
    ocr_tasks = [asyncio.create_task(extract_text_from_image(page)) for page in selected_pages]
    results = await asyncio.gather(face_task, *ocr_tasks)

    ocr_results = [text for text in results[1:] if text is not None]
    combined_text = " ".join(ocr_results)
    is_verified = verify_text(combined_text)

    return {
        "ocr_results": results[1:],
        "verification_passed": is_verified
    }
