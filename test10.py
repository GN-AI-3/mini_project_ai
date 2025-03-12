from diffusers import StableDiffusionImageVariationPipeline
from PIL import Image
import time  # 시간 측정을 위한 모듈 추가

# 시간 측정 시작
start_time = time.time()

# 파이프라인 모델 로드
pipe = StableDiffusionImageVariationPipeline.from_pretrained(
    "lambdalabs/sd-image-variations-diffusers", revision="v2.0"
)
pipe = pipe.to("cuda")

# 로컬 이미지 파일 경로
image_path = "business-person.png"  # 로컬 이미지 파일 경로

# 이미지 열기
image = Image.open(image_path).convert("RGB")

# 이미지 변형 생성 (1개의 이미지만 생성)
out = pipe(image, num_images_per_prompt=1, guidance_scale=15)

# 변형된 첫 번째 이미지 저장
out["images"][0].save("result.jpg")

# 시간 측정 종료
end_time = time.time()

# 걸린 시간 출력
elapsed_time = end_time - start_time
print(f"Time taken to process and save the image: {elapsed_time:.2f} seconds")
