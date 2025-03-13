import csv
import random

# CSV 파일 읽기
predefined_data = []
with open('predefined_titles_and_descriptions.csv', mode='r', encoding='utf-8') as file:
    reader = csv.reader(file)
    for row in reader:
        predefined_data.append(row)


def find_title_and_description(text):
    
    # 선정하는 로직 추가 예정
    
    random_num = random.randint(0, 97)
    selected = predefined_data[random_num]

    return selected[0], selected[2]

# # 사용 예시
# input_text = "이 학생은 리더십이 뛰어나며 팀워크를 중시합니다."
# keyword, description = find_title_and_description(input_text)

# print("제목:", keyword)
# print("설명글:", description)
