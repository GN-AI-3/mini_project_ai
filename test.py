import torch
from diffusers import StableDiffusionInstructPix2PixPipeline
from diffusers.utils import load_image
import time  # 시간 측정을 위한 모듈 추가

# 시간 측정 시작
start_time = time.time()

# 모델 로드
model_id = "instruction-tuning-sd/cartoonizer"
pipeline = StableDiffusionInstructPix2PixPipeline.from_pretrained(
    model_id, torch_dtype=torch.float16, use_auth_token=True  # float16으로 메모리 사용 줄이기
)

# 모델을 GPU로 이동
pipeline.to("cuda")

# 이미지 로드
image_path = "business-person.png"
image = load_image(image_path)

# 이미지 해상도 축소 (옵션)
image = image.resize((512, 512))  # 해상도 축소 (필요에 따라 크기 조정)

# 간단한 프롬프트로 변경
prompt = "Cartoonize this image"

# 이미지 변환
result = pipeline(prompt, image=image)
cartoon_image = result.images[0]

# 이미지 저장
cartoon_image.save("cartoon_images.png")

# 시간 측정 종료
end_time = time.time()

# 걸린 시간 출력
elapsed_time = end_time - start_time
print(f"Time taken to process and save the image: {elapsed_time:.2f} seconds")
