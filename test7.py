import torch
from diffusers import StableDiffusionPipeline
from PIL import Image
import time  # 시간을 측정하기 위한 모듈 추가

# 사용할 모델과 파이프라인 로드
model_name = "CompVis/stable-diffusion-v1-4-original"
pipe = StableDiffusionPipeline.from_pretrained(model_name)

# GPU가 사용 가능한지 확인하고, GPU로 모델을 이동
device = "cuda" if torch.cuda.is_available() else "cpu"
pipe.to(device)  # 모델을 GPU로 이동 (GPU가 없으면 CPU 사용)

# 이미지 로드
image_path = "business-person.png"  # 본인의 이미지 경로로 수정
image = Image.open(image_path).convert("RGB")  # 이미지가 RGBA일 경우 변환

# 변환할 스타일 프롬프트 설정
prompt = "Cartoonize this image"

# 이미지 변환에 걸린 시간 측정을 위한 시간 시작
start_time = time.time()

# 스타일 변환
result = pipe(prompt=prompt, init_image=image, strength=0.75).images[0]  # strength를 조정하여 얼마나 스타일을 적용할지 설정

# 이미지 변환에 걸린 시간 끝
end_time = time.time()

# 소요 시간 출력
elapsed_time = end_time - start_time
print(f"이미지 변환에 걸린 시간: {elapsed_time:.2f} 초")

# 변환된 이미지를 저장
result.save("styled_image.png")

# 변환된 이미지 출력
result.show()
