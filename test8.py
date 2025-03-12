import torch
from diffusers import StableDiffusionImg2ImgPipeline  # Img2Img 파이프라인 사용
from PIL import Image
import time  # 시간을 측정하기 위한 모듈 추가

# CartoonGAN 모델 로드 (Stable Diffusion Img2Img 파이프라인 사용)
model_id = "CompVis/stable-diffusion-v1-4-original"  # 모델 ID를 적절하게 수정
pipe = StableDiffusionImg2ImgPipeline.from_pretrained(model_id, torch_dtype=torch.float32)

# GPU가 사용 가능한지 확인하고, GPU로 모델을 이동
device = "cuda" if torch.cuda.is_available() else "cpu"
pipe.to(device)  # 모델을 GPU로 이동 (GPU가 없으면 CPU 사용)

# 이미지 로드
image_path = "business-person.png.jpg"  # 본인의 이미지 경로로 수정
image = Image.open(image_path).convert("RGB")  # 이미지를 RGB로 변환

# 변환할 스타일 프롬프트 설정 (카툰 스타일로 변환)
prompt = "Cartoonize this image, giving it a colorful, exaggerated cartoon appearance."

# 시간 측정 시작
start_time = time.time()

# 스타일 변환
result = pipe(prompt=prompt, init_image=image, strength=0.75).images[0]

# 시간 측정 끝
end_time = time.time()

# 소요 시간 출력
elapsed_time = end_time - start_time
print(f"이미지 변환에 걸린 시간: {elapsed_time:.2f} 초")

# 변환된 이미지 저장
result.save("cartoon_image.png")

# 변환된 이미지 출력
result.show()
