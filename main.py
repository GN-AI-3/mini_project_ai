from fastapi import FastAPI, UploadFile, File
from typing import List
from pykospacing import Spacing
from PIL import Image
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from transformers import pipeline
import os
import kss
import re
import io

import uuid
import asyncio
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
from google.cloud import vision
from pdf2image import convert_from_bytes

import insightface

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'mini-api-test-a0538c7dd495.json'

app = FastAPI()
executor = ThreadPoolExecutor(max_workers=4)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

####################################################################################################### 
# 모델 선언
# 모델은 전역에서 선언
#######################################################################################################

model = None

model_text_style = pipeline(
    'text2text-generation',
    model='heegyu/kobart-text-style-transfer'
)

vision_client = vision.ImageAnnotatorClient()

face_model = insightface.app.FaceAnalysis(providers=['CUDAExecutionProvider'])
face_model.prepare(ctx_id=0)

VERIFICATION_KEYWORDS = ["행동 특성 및 종합의견", "학교", "반"]

#######################################################################################################
# API 설정
# 
#######################################################################################################

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

# return 예시
# return JSONResponse(
#         content={
#             "message": "이미지 분류 요청이 성공적으로 처리되었습니다",
#             "image_filename": image.filename,
#             "top_category": top_category.category_name,
#             "score": float(top_category.score)  # float32를 JSON 직렬화 가능한 형태로 변환
#         },
#         status_code=200
#     )

@app.post("/test")
async def test(
    file: UploadFile = File(...)
):
    try:
        os.makedirs("files", exist_ok=True)
        file_location = f"files/{file.filename}"
        with open(file_location, "wb") as f:
            f.write(await file.read())

        return JSONResponse(
            content={
                "message": "파일 업로드 성공!",
                "file_name": file.filename
            },
            status_code=200
        )
    
    except Exception as e:
        return JSONResponse(
            content={
                "message": f"파일 업로드 실패: {str(e)}"
            },
            status_code=500
        )

#######################################################################################################
# 함수 정의
# 정의 된 함수를 /pdf_process 에서 호출
#######################################################################################################

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

    os.makedirs(save_dir, exist_ok=True)

    unique_id = uuid.uuid4().hex
    jpg_path = os.path.join(save_dir, f"face_{unique_id}.jpg")
    png_path = os.path.join(save_dir, f"face_{unique_id}.png")

    cv2.imwrite(jpg_path, face_crop_bgr)
    cv2.imwrite(png_path, face_crop_bgr)

    return jpg_path, png_path


# 추출된 텍스트 문장 단위로 구분 및 불필요한 문자 제거
def text_split(
    text: str
):
    processed_text = text.replace("\n", " ").strip()
    processed_text = processed_text.replace("gov.kr", "").replace("정부24", "").replace("OCR Result for Page : ", "").replace("KOR", "")
    processed_text = re.sub(r'문서확인번호: .+? \(신청인 : .+?\)', '', processed_text)
    processed_text = re.sub(r'\S+학교 .*?년 .*?월 .*?일\s*.*?/.*?\s*반\s*.*?\s*번호\s*.*?\s*이름\s*\S+', '', processed_text)
    processed_text = re.sub(r'\b행동 특성 및 종합의견\b', '', processed_text)

    processed_text = re.sub(r'학교 .+? \(신청인 : .+?\)', '', processed_text)
    processed_text = re.sub(r'\b학년\b', '', processed_text)
    processed_text = re.sub(r'\b\d+\b', '', processed_text)
    processed_text = processed_text.replace(" ", "")

    try:
        spacing = Spacing()
        corrected_text = spacing(processed_text)
    except Exception as e:
        print(f"띄어쓰기 교정 중 오류 발생: {e}")
        corrected_text = processed_text

    sentences = kss.split_sentences(corrected_text)
    return [sentence.strip() for sentence in sentences if sentence.strip()]


# 문장 단위로 구분된 텍스트 장/단점 구분
def text_prosCons(
    text: List[str]
):
    pass


# 장/단점 구분된 텍스트 요약
def summarizeProsCons(
    prosCons: list[str]
):
    pass


# 요약된 텍스트 구어체로 변경
def text_to_speech(
    prosCons: list[str]
):
    # styles = ['구어체','안드로이드','아재','채팅',
    # '초등학생','이모티콘','enfp','신사','할아버지','할머니','중학생',
    # '왕','나루토','선비','소심한','번역기']

    style = ['구어체']
    converted_texts = []

    for txt in prosCons:
        input_text = f"{style} 말투로 변환:{txt}"
        out = model_text_style(input_text, max_length=128)
        converted_texts.append(out[0]['generated_text'])
    return converted_texts


# 이미지 카툰화
def image_process(
    image_path: str,
    image_bytes: bytes
):
    image = image_process1(image_path)
    image = image_process2(image_bytes)
    pass


# 요약된 장/단점과 카툰화된 이미지를 하나의 이미지로 생성
def get_image(
    image_path: str,
    image_bytes: bytes,
    prosCons: list[str]
):
    image = image_process1(image_path)
    image = image_process2(image_bytes)
    pass