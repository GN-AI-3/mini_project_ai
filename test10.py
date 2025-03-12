import torch
from PIL import Image
from diffusers import StableDiffusionImg2ImgPipeline
import time

# 모델 로딩 (Stable Diffusion Inpainting 또는 Img2Img 파이프라인)
pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float32  # float16 대신 float32 사용
)
pipe.to("cuda")  # CUDA(GPU)가 있다면 이를 사용하여 속도를 향상시킴

# 이미지 불러오기 (로컬 이미지 파일)
def load_image(image_path):
    return Image.open(image_path).convert("RGB")

# 이미지 변환 함수
def convert_to_cartoon(image_path, prompt="A cartoon version of a business person", strength=0.8):
    # 이미지 로드
    image = load_image(image_path)
    
    # 이미지 리사이즈 (Stable Diffusion은 512x512 크기를 권장)
    image = image.resize((512, 512))

    # 이미지 변환
    # 반드시 image가 PIL.Image.Image 형식인지 확인
    if not isinstance(image, Image.Image):
        raise TypeError(f"Expected a PIL.Image.Image object, but got {type(image)}")
    
    # 'images'는 결과의 리스트입니다. 첫 번째 이미지를 가져옵니다.
    result = pipe(prompt=prompt, image=image, strength=strength).images[0]
    
    return result

# 시간 측정 시작
start_time = time.time()

# 변환할 이미지 경로 (변환할 이미지 파일의 경로)
image_path = "business-person.png"  # 여기 경로에 변환할 이미지를 넣으세요

# 카툰 스타일로 변환
cartoon_image = convert_to_cartoon(image_path)

# 결과 이미지 저장
cartoon_image.save("cartoon_image_output.png")

# 시간 측정 끝
end_time = time.time()
elapsed_time = end_time - start_time

# 걸린 시간 출력
print(f"Time taken to process the image: {elapsed_time:.2f} seconds")

# 변환된 이미지 보기
cartoon_image.show()
