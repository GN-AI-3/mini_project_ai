import torch
from diffusers import AutoPipelineForImage2Image
from PIL import Image
import time  # 시간을 측정하기 위한 모듈 추가

# 모델 로드
pipeline = AutoPipelineForImage2Image.from_pretrained(
    "stable-diffusion-v1-5/stable-diffusion-v1-5", torch_dtype=torch.float16, variant="fp16", use_safetensors=True
)

# GPU로 모델을 강제 이동
pipeline.to("cuda")

# 모델의 장치 확인
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"사용 중인 장치: {device}")

# 로컬 이미지 로드
init_image = Image.open("business-person.png")  # 로컬 파일 경로로 이미지 로드

# 프롬프트 설정 (카툰 스타일 변환)
# prompt = "Cartoonize this image in a colorful and fun style, with exaggerated features and bright colors."
prompt = "Cartoonize this image"
# 시간 측정 시작
start_time = time.time()

# 모델을 사용하여 이미지 변환
image = pipeline(prompt, image=init_image).images[0]

# 시간 측정 끝
end_time = time.time()

# 소요 시간 출력
elapsed_time = end_time - start_time
print(f"이미지 변환에 걸린 시간: {elapsed_time:.2f} 초")

# 변환된 이미지만 저장
image.save("cartoonized_image.png")

print("변환된 카툰 이미지가 'cartoonized_image.png'로 저장되었습니다.")
