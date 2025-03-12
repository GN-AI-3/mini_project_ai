from openai import OpenAI
import os
from dotenv import load_dotenv

# .env 파일에서 환경 변수 로드
load_dotenv()

def convert_to_casual_style(text):
    # OpenAI API 키 가져오기
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        raise ValueError("OPENAI_API_KEY가 설정되지 않았습니다. .env 파일을 확인해주세요.")
    
    # OpenAI 클라이언트 초기화
    client = OpenAI(api_key=api_key)

    # 프롬프트 구성
    prompt = f"""다음 문장을 자연스러운 구어체로 바꿔주세요. 
원문의 의미는 유지하되, 친근하고 부드러운 말투로 변환해주세요.
단, 존댓말은 유지해주세요.

원문: {text}
변환:"""
    
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        store=True,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    return completion.choices[0].message;
    
    # # API 호출
    # response = client.chat.completions.create(
    #     model="gpt-3.5-turbo",
    #     messages=[
    #         {"role": "system", "content": "당신은 문어체를 자연스러운 구어체로 변환해주는 전문가입니다. 항상 원문의 의미는 유지하면서, 더 친근하고 부드러운 말투로 변환합니다."},
    #         {"role": "user", "content": prompt}
    #     ],
    #     temperature=0.7,
    #     max_tokens=150
    # )
    
    # # 결과 반환
    # return response.choices[0].message.content.strip()

# 테스트
if __name__ == "__main__":
    test_texts = [
        "학생의 성실성과 책임감이 돋보입니다."
    ]
    
    print("=== 구어체 변환 테스트 ===")
    for text in test_texts:
        print("\n원문:", text)
        try:
            print("변환:", convert_to_casual_style(text))
        except Exception as e:
            print("오류 발생:", str(e))
