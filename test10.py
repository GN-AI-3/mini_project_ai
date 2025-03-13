import torch
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse
from io import BytesIO
from PIL import Image, ImageDraw, ImageFont
from diffusers import StableDiffusionImg2ImgPipeline, StableDiffusionPipeline
import numpy as np
import cv2
import mediapipe as mp
import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# FastAPI 설정
app = FastAPI()

# Stable Diffusion 모델 로딩
pipe = StableDiffusionImg2ImgPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16)
pipe.to("cuda")  # CUDA(GPU)가 있다면 이를 사용하여 속도를 향상시킴

# 모델 로드: 이미지 생성 stable-diffusion-v1-5 모델 사용 (torch.float16으로 설정하여 속도 향상)
makeImage = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16)
makeImage.to("cuda") 

# MediaPipe Selfie Segmentation 설정
mp_selfie_segmentation = mp.solutions.selfie_segmentation

# 만화 스타일 변환 함수
def convert_to_cartoon(image: Image, prompt="Randomly transformed cartoon image with unique features and playful details.", strength=0.6, guidance_scale=8, num_inference_steps=50):
    image = image.resize((512, 512))
    result = pipe(prompt=prompt, image=image, strength=strength, guidance_scale=guidance_scale, num_inference_steps=num_inference_steps).images[0]
    return result

# 텍스트 추가 함수
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

def apply_new_background(input_image, background_image):
    image_rgb = cv2.cvtColor(np.array(input_image), cv2.COLOR_RGB2BGR)
    with mp_selfie_segmentation.SelfieSegmentation(model_selection=1) as selfie_segmentation:
        result = selfie_segmentation.process(image_rgb)
        condition = result.segmentation_mask > 0.5  # 사람 부분만 추출
        image_no_bg = image_rgb.copy()
        image_no_bg[~condition] = 255  # 배경을 흰색으로 설정
        background_resized = cv2.cvtColor(np.array(background_image), cv2.COLOR_RGB2BGR)
        new_background_resized = cv2.resize(background_resized, (image_rgb.shape[1], image_rgb.shape[0]))
        final_image = np.where(condition[:, :, None], image_rgb, new_background_resized)
        final_image_pil = Image.fromarray(cv2.cvtColor(final_image, cv2.COLOR_BGR2RGB))
        return final_image_pil

# 비동기 이미지 생성 함수
async def create_background():
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
    
    # 저장할 경로 설정
    image_path = "generated_background.png"
    background.save(image_path)
    return image_path

@app.post("/process_image")
async def process_image(image1: UploadFile = File(...)):
    plantext="장점을 요약한 겁니다."
    text = "테스트입니다. 테스트입니다. 테스트입니다. 테스트."
    
    # # 업로드된 이미지를 비동기적으로 읽어옴
    image_data = await image1.read()  # await를 사용하여 비동기적으로 읽음
    input_image = Image.open(BytesIO(image_data))  # 업로드된 이미지를 PIL 객체로 변환

    # 1. 배경 생성과 만화 변환을 동기로 진행
    background_path = await create_background()  # 배경 생성 후 경로 반환
    background = Image.open(background_path)  # 경로로부터 배경 이미지 열기
    cartoon_image = convert_to_cartoon(input_image)  # 만화 스타일 변환

    # background는 이미 PIL Image 객체이므로 Image.open() 호출하지 않음
    background = background.convert("RGB")  # RGB로 변환
    background = background.resize((512, 512))  # 배경 크기 변경
        # # 3. MediaPipe로 사람을 구분하고 배경을 합성
    cartoon_image_with_text = apply_new_background(cartoon_image, background)

    final_image = add_text_below_image(cartoon_image_with_text, text=text,plantext=plantext)
    
    # 4. 최종 이미지를 메모리로 처리
    img_byte_arr = BytesIO()
    final_image.save(img_byte_arr, format='PNG')  # PNG 형식으로 저장 (투명도 지원)
    img_byte_arr.seek(0)

    # 5. 최종 결과를 StreamingResponse로 반환
    return StreamingResponse(img_byte_arr, media_type="image/png")
