import time
import torch
from diffusers import StableDiffusionPipeline

# 모델 로드
repo_id = "lavaman131/cartoonify"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch_dtype = torch.float16 if device.type in ["mps", "cuda"] else torch.float32

pipeline = StableDiffusionPipeline.from_pretrained(repo_id, torch_dtype=torch_dtype).to(device)

# 새로운 이미지 생성 프롬프트 설정
prompt = "A cartoon of a businesswoman in a modern office."

# 시간 측정 시작
start_time = time.time()

# 이미지 생성
image = pipeline(prompt).images[0]

# 시간 측정 끝
end_time = time.time()

# 이미지 저장
image.save("generated_cartoon_image.png")

# 소요 시간 출력
elapsed_time = end_time - start_time
print(f"이미지 생성에 걸린 시간: {elapsed_time:.2f} 초")
