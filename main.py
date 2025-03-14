from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse
from typing import List
from pykospacing import Spacing
from PIL import Image, ImageDraw, ImageFont
from fastapi.middleware.cors import CORSMiddleware  # 한 번만 임포트
from fastapi.responses import JSONResponse
from transformers import pipeline
from sentence_transformers import SentenceTransformer, util
import time
import pickle
import os
import kss
import re
import io
import torch
import mediapipe as mp

# student_evaluation.py 모듈 임포트
import student_evaluation as se
from io import BytesIO
import uuid
import asyncio
from concurrent.futures import ThreadPoolExecutor
import cv2
import numpy as np
from google.cloud import vision
from pdf2image import convert_from_bytes
from diffusers import StableDiffusionImg2ImgPipeline, StableDiffusionPipeline
import insightface

####################################################################################################### 
# 전역 변수 선언
#
#######################################################################################################
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'mini-api-test-a0538c7dd495.json'
PREDEFINED_OUTPUT_CSV_FILE_PATH = "predefined_titles_and_descriptions.csv"
EMBEDDING_DATA_FILE_PATH = "embeddings.pkl"
VERIFICATION_KEYWORDS = ["행동 특성 및 종합의견", "학교", "반"]

# 모델 로드 (전역 변수)
embedding_model_instance = None
spacing_instance = Spacing()

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

model_text_style = pipeline(
    'text2text-generation',
    model='heegyu/kobart-text-style-transfer'
)

vision_client = vision.ImageAnnotatorClient()

face_model = insightface.app.FaceAnalysis(providers=['CUDAExecutionProvider'])
face_model.prepare(ctx_id=0)

embedding_model_instance = SentenceTransformer('sentence-transformers/paraphrase-multilingual-mpnet-base-v2')

# Stable Diffusion 모델 로딩
pipe = StableDiffusionImg2ImgPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16)
pipe.to("cuda")  # CUDA(GPU)가 있다면 이를 사용하여 속도를 향상시킴

# 모델 로드: 이미지 생성 stable-diffusion-v1-5 모델 사용 (torch.float16으로 설정하여 속도 향상)
makeImage = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16)
makeImage.to("cuda") 

# MediaPipe Selfie Segmentation 설정
mp_selfie_segmentation = mp.solutions.selfie_segmentation
#######################################################################################################
# API 설정
# 
#######################################################################################################

@app.post("/process-pdf/")
async def process_pdf(
    file: UploadFile = File(...)
):
    start_time = time.time()
    final_start_time = time.time()
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
    end_time = time.time()
    print("OCR 결과 : " + combined_text)
    print(f"PDF 처리 시간: {end_time - start_time}초")

    if is_verified == False:
        return {"error": "Not a school record."}
    
    try:
        # 추출된 텍스트를 문장 단위로 분리
        start_time = time.time()
        sentences = text_split(combined_text)
        end_time = time.time()
        print("텍스트 분리 결과:", sentences)  # 리스트를 직접 출력
        print(f"텍스트 분리 시간: {end_time - start_time}초")
        
        # 문장을 장/단점으로 분석
        start_time = time.time()
        analysis_result = text_prosCons(sentences)
        end_time = time.time()
        print("장/단점 분석 결과:", analysis_result)
        print(f"장/단점 분석 시간: {end_time - start_time}초")

        # 장/단점 요약
        start_time = time.time()
        summarizeProsCon = get_most_similar_sentence(analysis_result[0])
        end_time = time.time()
        print("장/단점 요약 결과:", summarizeProsCon)
        print(f"장/단점 요약 시간: {end_time - start_time}초")

        # 이미지 변경  
        start_time = time.time()
        face_jpg_path, face_png_path = face_task  # 튜플 언패킹
        background = await create_background(os.path.dirname(face_jpg_path))  # 배경 생성
        img = get_image(image_filename=face_jpg_path, background=background, text=summarizeProsCon, plantext=analysis_result)

        # 이미지를 바이트 스트림으로 변환
        img_byte_arr = BytesIO()
        img.save(img_byte_arr, format="PNG")
        img_byte_arr.seek(0)  # 스트림 포인터를 처음으로 이동
        
        # 이미지 스트리밍 반환
        response = StreamingResponse(img_byte_arr, media_type="image/png")
        end_time = time.time()
        print(f"이미지 처리 시간: {end_time - start_time}초")
        final_end_time = time.time()
        print(f"최종 처리 시간: {final_end_time - final_start_time}초")

        return JSONResponse(
            content={
                "message": "PDF 처리 및 분석이 성공적으로 완료되었습니다",
                "advantages": analysis_result["장점"],
                "disadvantages": analysis_result["단점"],
                "response": response
            },
            status_code=200
        )
    except Exception as e:
        return JSONResponse(
            content={
                "message": f"PDF 처리 중 오류 발생: {str(e)}"
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
    processed_text = remove_before_keyword(text, "행동 특성 및 종합의견")
    processed_text = text.replace("\n", " ").strip()
    processed_text = processed_text.replace("gov.kr", "").replace("정부24", "").replace("OCR Result for Page : ", "").replace("KOR", "")
    processed_text = re.sub(r'문서확인번호 .+? \(신청인 : .+?\)', '', processed_text)
    processed_text = re.sub(r'문서 확인번호 .+? \(신청인 : .+?\)', '', processed_text)
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

def remove_before_keyword(text: str, keyword: str) -> str:
    index = text.find(keyword)
    if index != -1:  
        return text[index:]
    return text

# 문장 단위로 구분된 텍스트 장/단점 구분
def text_prosCons(
    text: List[str]
):
    max_items = 5
    
    try:
        # 전달된 텍스트 분석
        analysis_result = se.analyze_student_evaluation(text, max_items=max_items)
        result = analysis_result.get("장점")
        return result
    except Exception as e:
        print(f"텍스트 분석 중 오류 발생: {e}")
        # 오류 발생 시 빈 결과 반환
        return {"장점": [], "단점": []}


# 장/단점 구분된 텍스트 요약
def get_most_similar_sentence(
        input_sentence: str
    ):
    """
    주어진 문장과 저장된 임베딩 데이터를 활용해 가장 유사한 문장을 찾는 함수.

    Args:
        input_sentence (str): 유사도를 비교할 입력 문장.

    Returns:
        tuple: 유사도 점수, 가장 유사한 문장의 제목, 설명
               (유사도 점수, 제목, 설명).
    """
    # 임베딩 데이터와 CSV 데이터를 로드
    try:
        with open(EMBEDDING_DATA_FILE_PATH, 'rb') as embedding_file:
            csv_data_rows, embedding_vectors = pickle.load(embedding_file)
    except FileNotFoundError:
        raise ValueError(f"임베딩 파일 {EMBEDDING_DATA_FILE_PATH}을(를) 찾을 수 없습니다.")
    except pickle.UnpicklingError:
        raise ValueError(f"임베딩 파일 {EMBEDDING_DATA_FILE_PATH}의 형식이 올바르지 않습니다.")

    # 입력 문장 전처리 및 임베딩 생성
    input_sentence_embedding = embedding_model_instance.encode(input_sentence)

    # 코사인 유사도 계산
    similarity_scores = util.cos_sim(input_sentence_embedding, embedding_vectors)
    most_similar_index = np.argmax(similarity_scores)
    most_similar_score = similarity_scores[0][most_similar_index]

    # 가장 유사한 행 데이터 가져오기
    matched_row = csv_data_rows[most_similar_index]
    matched_title = matched_row[0]
    matched_description = matched_row[2]

    return matched_title, matched_description, most_similar_score.item()


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
def image_process(image_path: str, prompt="Randomly transformed cartoon image with unique features and playful details.", strength=0.6, guidance_scale=8, num_inference_steps=50):
    image = Image.open(image_path)
    image = image.resize((512, 512)) 

    result = pipe(prompt=prompt, image=image, strength=strength, guidance_scale=guidance_scale, num_inference_steps=num_inference_steps).images[0]
    
    return result

# 요약된 장/단점과 카툰화된 이미지를 하나의 이미지로 생성
def get_image(
    image_path: str,
    background: str,
    text: str,
    plantext: str
):
    apply_background = apply_new_background(image_path, background)
    
    # 4. 텍스트를 이미지 하단에 추가
    cartoon_image_with_text = add_text_below_image(apply_background, text,plantext)
    return cartoon_image_with_text


# 텍스트를 최대 너비에 맞춰 분할하는 함수
def wrap_text(draw, font, text, max_width):
    lines = []
    words = text.split(' ')
    current_line = ""
    for word in words:
        test_line = current_line + " " + word if current_line else word
        test_bbox = draw.textbbox((0, 0), test_line, font=font)
        test_width = test_bbox[2] - test_bbox[0]
        if test_width <= max_width:
            current_line = test_line
        else:
            lines.append(current_line)
            current_line = word
    if current_line:
        lines.append(current_line)
    return lines

# 택스트 추가 이미지 생성
def add_text_below_image(input_image, text, plantext):
    img = input_image
    try:
        font = ImageFont.truetype("fonts/malgun.ttf", 30)  # 한글 폰트 경로
    except IOError:
        font = ImageFont.load_default()  # 기본 폰트

    width, height = img.size
    draw = ImageDraw.Draw(img)

    # 텍스트를 이미지 상단에 넣기 위한 준비
    max_width = width * 0.7  # 양옆에 여유를 두기 위해 120%로 설정
    plan_lines = wrap_text(draw, font, plantext, max_width)

    # 상단 텍스트의 높이 계산
    plan_line_height = 0
    for line in plan_lines:
        text_bbox = draw.textbbox((0, 0), line, font=font)
        line_height = text_bbox[3] - text_bbox[1]
        plan_line_height += line_height
    plan_line_height += 20  # 줄 간격 추가

    # 기존 텍스트를 아래에 추가하는 계산
    max_width = width * 1.2 
    lines = wrap_text(draw, font, text, max_width)
    line_spacing = 20
    total_text_height = 0
    for line in lines:
        text_bbox = draw.textbbox((0, 0), line, font=font)
        line_height = text_bbox[3] - text_bbox[1]
        total_text_height += line_height
    total_text_height += line_spacing * (len(lines) - 1)  # 줄 간격 추가

    # 새로운 높이 계산 (상단 텍스트 + 기존 텍스트)
    new_height = height + plan_line_height + total_text_height + 20  # 텍스트를 위한 공간 + 20px 여유
    new_img = Image.new('RGB', (width, new_height), color='white')
    new_img.paste(img, (0, 0))

    # 이미지에 텍스트 추가
    draw = ImageDraw.Draw(new_img)
    # plantext 상단 텍스트
    text_y = 10  # 상단에 넣기 위해 y를 10px로 설정
    for line in plan_lines:
        text_bbox = draw.textbbox((0, 0), line, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_x = (width - text_width) // 2
        draw.text((text_x, text_y), line, font=font, fill='black')
        text_y += text_bbox[3] - text_bbox[1] + line_spacing

    # 기존 text 하단 텍스트
    text_y = height + plan_line_height + 10  # 기존 이미지 하단부터 시작
    for line in lines:
        text_bbox = draw.textbbox((0, 0), line, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_x = (width - text_width) // 2
        draw.text((text_x, text_y), line, font=font, fill='black')
        text_y += text_bbox[3] - text_bbox[1] + line_spacing

    return new_img

# 사람 특정해서 배경 변경 
def apply_new_background(input_image_path, background_image_path):
    # 입력 이미지와 배경 이미지를 파일 경로로 열기
    input_image = Image.open(input_image_path).convert("RGB")  # RGB 모드로 열기
    background_image = Image.open(background_image_path).convert("RGB")  # RGB 모드로 열기
    # PIL 이미지를 NumPy 배열로 변환 (OpenCV에서 사용하기 위해)
    image_rgb = cv2.cvtColor(np.array(input_image), cv2.COLOR_RGB2BGR)
    with mp_selfie_segmentation.SelfieSegmentation(model_selection=1) as selfie_segmentation:
        # 사람을 구분하기 위한 Segmentation 처리
        result = selfie_segmentation.process(image_rgb)
        condition = result.segmentation_mask > 0.5  # 사람 부분만 추출
        # 배경을 흰색으로 설정
        image_no_bg = image_rgb.copy()
        image_no_bg[~condition] = 255  # 배경을 흰색으로
        # 배경 이미지를 OpenCV 형태로 변환
        background_resized = cv2.cvtColor(np.array(background_image), cv2.COLOR_RGB2BGR)
        new_background_resized = cv2.resize(background_resized, (image_rgb.shape[1], image_rgb.shape[0]))
        # 사람 부분은 원본 이미지에서, 배경 부분은 새로운 배경으로 설정
        final_image = np.where(condition[:, :, None], image_rgb, new_background_resized)
        # 결과 이미지를 PIL 객체로 변환
        final_image_pil = Image.fromarray(cv2.cvtColor(final_image, cv2.COLOR_BGR2RGB))
        return final_image_pil

# 비동기 이미지 생성 함수
#이미지 경로받기
async def create_background(background_image_path):
    img_path = os.path.join(background_image_path, "back")  # 배경 이미지 경로
    prompt = "Simple background"
    dynamic_seed = torch.randint(0, 2**32, (1,), dtype=torch.int64).item()  # 동적 시드 생성
    
    # 비동기적으로 이미지를 생성
    background = makeImage(
        prompt,
        height=512,
        width=512,
        guidance_scale=6.0,
        num_inference_steps=20,
        generator=torch.Generator("cpu").manual_seed(dynamic_seed)  # 동적 시드를 설정
    ).images[0]
    
    background = background.convert("RGB")  # RGB로 변환
    background = background.resize((512, 512))  # 배경 크기 변경
    background.save(img_path)  # 이미지 저장
    
    return img_path  # 생성된 이미지 경로 리턴
def delete_image(image_filename, background):
    try:
        if os.path.exists(image_filename):
            os.remove(image_filename)
            print(f"{image_filename} 파일이 삭제되었습니다.")
        else:
            print(f"{image_filename} 파일이 존재하지 않습니다.")
    except Exception as e:
        print(f"파일 삭제 중 오류가 발생했습니다: {e}")

    try:
        if os.path.exists(background):
            os.remove(background)
            print(f"{background} 파일이 삭제되었습니다.")
        else:
            print(f"{background} 파일이 존재하지 않습니다.")
    except Exception as e:
        print(f"배경 파일 삭제 중 오류가 발생했습니다: {e}")