from diffusers import StableDiffusionImg2ImgPipeline
from PIL import Image
import torch
import time

# 시간 측정 시작
start_time = time.time()

# 파이프라인 모델 로드 (Stable Diffusion)
pipe = StableDiffusionImg2ImgPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float32)
pipe = pipe.to("cuda")

# 로컬 이미지 파일 경로
image_path = "business-person.png"  # 본인의 이미지 경로로 수정

# 이미지 열기
image = Image.open(image_path).convert("RGB")  # RGB로 변환하여 올바른 형식으로 만듦

# 프롬프트 설정 (카툰화)
prompt = "Cartoonize this image"

# 이미지 변환
result = pipe(prompt=prompt, init_image=image, strength=0.75).images[0]

# 변환된 이미지 저장
result.save("cartoon_image.png")

# 시간 측정 끝
end_time = time.time()
elapsed_time = end_time - start_time

# 걸린 시간 출력
print(f"Time taken to process the image: {elapsed_time:.2f} seconds")

# 변환된 이미지 보기
result.show()
