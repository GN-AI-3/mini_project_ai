from fastapi import FastAPI, UploadFile, File
from typing import List
from pykospacing import Spacing
from PIL import Image, ImageDraw, ImageFont
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import os
import kss
import re
import io
import torch
from fastapi.responses import StreamingResponse
from io import BytesIO
from diffusers import StableDiffusionImg2ImgPipeline, StableDiffusionPipeline
import numpy as np
import cv2
import mediapipe as mp
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Stable Diffusion 모델 로딩
pipe = StableDiffusionImg2ImgPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16)
pipe.to("cuda")  # CUDA(GPU)가 있다면 이를 사용하여 속도를 향상시킴
 
# 모델 로드: 이미지 생성 stable-diffusion-v1-5 모델 사용 (torch.float16으로 설정하여 속도 향상)
makeImage = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16)
makeImage.to("cuda") 
# MediaPipe Selfie Segmentation 설정
mp_selfie_segmentation = mp.solutions.selfie_segmentation


####################################################################################################### 
# 모델 선언
# 모델은 전역에서 선언
#######################################################################################################

model = None


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

        image_filename="user.png"
        background = await create_background(os.path.dirname(image_filename))  # 배경 생성
        img=get_image(image_filename=image_filename,background=background,text="테스트입니다. 테스트입니다. 테스트입니다.",plantext="장점입니다.")
        # 이미지를 바이트 스트림으로 변환
        img_byte_arr = BytesIO()
        img.save(img_byte_arr, format="PNG")
        img_byte_arr.seek(0)  # 스트림 포인터를 처음으로 이동
        # 이미지 스트리밍 반환
        response = StreamingResponse(img_byte_arr, media_type="image/png")

        delete_image(image_filename, background) 

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

    try:
        spacing = Spacing()
        corrected_text = spacing(processed_text)
    except Exception as e:
        print(f"띄어쓰기 교정 중 오류 발생: {e}")
        corrected_text = processed_text

    sentences = kss.split_sentences(corrected_text)
    return [sentence.strip() for sentence in sentences if sentence.strip()]

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
    max_items = 5
    
    try:
        # 전달된 텍스트 분석
        analysis_result = se.analyze_student_text(text, max_items=max_items)
        return analysis_result
    except Exception as e:
        print(f"텍스트 분석 중 오류 발생: {e}")
        # 오류 발생 시 빈 결과 반환
        return {"장점": [], "단점": []}


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
def image_process(image_path: str, prompt="Randomly transformed cartoon image with unique features and playful details.", strength=0.6, guidance_scale=8, num_inference_steps=50):
    # 이미지 파일을 열어 PIL 이미지 객체로 변환
    image = Image.open(image_path)
    image = image.resize((512, 512))  # 모델에 맞는 크기로 리사이즈
    
    # 카툰화 처리
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
