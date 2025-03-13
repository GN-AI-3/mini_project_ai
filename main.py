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

app = FastAPI()

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

#######################################################################################################
# API 설정
# 
#######################################################################################################

@app.post("/pdf_process")
async def pdf_to_image(
    pdf: UploadFile = File(...)
):
    pass

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

# 이미지 처리 방법 1
def image_process1(image_path: str):
    image = Image.open(image_path)  # 이미지 로드
    return image

# # 예시 사용
# image = image_process1("sample.jpg")
# image.show()

# 이미지 처리 방법 2
def image_process2(image_bytes: bytes):
    image = Image.open(io.BytesIO(image_bytes))  # 이미지 로드
    return image

# # 예시 사용
# with open("sample.jpg", "rb") as f:
#     image_data = f.read()

# image = image_process2(image_data)
# image.show()

# PDF 파일 OCR 처리
def pdf_process(
    pdf: UploadFile = File(...)
):
    # async 처리 OCR

    # async 처리 첫 페이지에서 이미지 추출

    pass


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