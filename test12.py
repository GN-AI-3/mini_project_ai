import time
from diffusers import StableDiffusionInpaintPipeline
from PIL import Image
import torch

# 로컬 이미지 로드
image_path = "business-person.png"  # 여기에 로컬 이미지 파일 경로를 입력하세요
image = Image.open(image_path)

# 모델 로드
pipe = StableDiffusionInpaintPipeline.from_pretrained(
    "sd-legacy/stable-diffusion-inpainting",
    revision="fp16",
    torch_dtype=torch.float16,
)

# 모델을 GPU로 이동
pipe.to("cuda")

# 카툰 스타일 프롬프트
prompt = "Cartoon style of a yellow cat, high resolution"

# 시간 측정 시작
start_time = time.time()

# 마스크 없이 이미지 변환
result_image = pipe(prompt=prompt, image=image).images[0]

# 시간 측정 끝
end_time = time.time()

# 결과 저장
result_image.save("./test12.png")

# 소요 시간 출력
elapsed_time = end_time - start_time
print(f"이미지 변환에 걸린 시간: {elapsed_time:.2f} 초")
