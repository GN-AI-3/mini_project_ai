# import fitz  # PyMuPDF의 별칭

# # PDF 파일 경로
# pdf_file = "doc.pdf"

# # PDF 열기
# pdf_document = fitz.open(pdf_file)

# # 고화질 이미지 생성
# for page_num in range(len(pdf_document)):
#     page = pdf_document[page_num]
#     # 해상도 설정: 300 DPI
#     pix = page.get_pixmap(dpi=300)  # DPI 값을 높여 화질 향상
#     pix.save(f"page_{page_num + 1}.png")  # PNG 포맷으로 저장


# import easyocr

# reader = easyocr.Reader(['ko'])
# result = reader.readtext('crop2.png')

# for row in result:
#     print(f"{row[0]}\n")

import cv2
import pytesseract

# 이미지 로드
image_path = "page_2.png"
image = cv2.imread(image_path)

# 그레이스케일로 변환
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 경계선 강조 (Thresholding)
_, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)

# 윤곽선 탐지
contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 표 영역 자르기
title_height = 80
table_count = 0
for contour in contours:
    # 윤곽선의 경계 상자 가져오기
    x, y, w, h = cv2.boundingRect(contour)
    y -= title_height
    h += title_height

    # 크기가 너무 작은 영역은 무시
    if w > 1700:
        print(w)
        table_count += 1
        cropped_table = image[y:y+h, x:x+w]
        
        # 잘라낸 표 이미지 저장
        cv2.imwrite(f"table_{table_count}.png", cropped_table)
