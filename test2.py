import cv2
import numpy as np
import matplotlib.pyplot as plt
import time

# 이미지 로드
image_path = "business-person.png"
img = cv2.imread(image_path)

# 시간 측정 시작
start_time = time.time()

# 그레이스케일로 변환 및 블러링
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = cv2.medianBlur(gray, 5)

# 에지 검출
edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9)

# 색상 부드럽게 하기
color = cv2.bilateralFilter(img, 9, 250, 250)

# 에지와 색상 결합
cartoon = cv2.bitwise_and(color, color, mask=edges)

# 시간 측정 끝
end_time = time.time()

# 결과 출력
plt.figure(figsize=(10, 10))
plt.imshow(cv2.cvtColor(cartoon, cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.title("Cartoon Image")
plt.show()

# 결과 저장
cv2.imwrite("cartoon_image.jpg", cartoon)

# 소요 시간 출력
elapsed_time = end_time - start_time
print(f"카툰 효과 변환에 걸린 시간: {elapsed_time:.2f} 초")
