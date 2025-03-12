import cv2
import numpy as np
from sklearn.cluster import KMeans
import time  # 시간을 측정하기 위한 모듈 추가

def cartoonify_image(image_path):
    # 이미지 로드
    img = cv2.imread(image_path)

    # HSV 색상 공간으로 변환
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # k-means 클러스터링
    kmeans = KMeans(n_clusters=5)
    kmeans.fit(hsv.reshape(-1, 3))
    quantized_hsv = kmeans.cluster_centers_[kmeans.labels_].astype(np.uint8)
    quantized_hsv = quantized_hsv.reshape(hsv.shape)

    # 히스토그램 평활화
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    # 에지 검출
    edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9)

    # 에지와 색상 결합
    cartoon = cv2.bitwise_and(cv2.cvtColor(quantized_hsv, cv2.COLOR_HSV2BGR), cv2.cvtColor(quantized_hsv, cv2.COLOR_HSV2BGR), mask=edges)

    return cartoon

# 이미지 경로
image_path = "business-person.png"

# 시간 측정 시작
start_time = time.time()

# 카툰 변환
cartoon_image = cartoonify_image(image_path)

# 시간 측정 끝
end_time = time.time()

# 결과 저장
cv2.imwrite("cartoon_image1.jpg", cartoon_image)

# 소요 시간 출력
elapsed_time = end_time - start_time
print(f"카툰화 처리에 걸린 시간: {elapsed_time:.2f} 초")
