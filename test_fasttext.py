"""
student_evaluation.py 모듈 테스트 파일
"""

import student_evaluation as se

def test_basic_analysis():
    """기본 학생 평가 텍스트 분석 테스트"""
    se.analyze_student_evaluation(max_items=5)  # 기본 예제 사용, 상위 5개 출력

if __name__ == "__main__":
    test_basic_analysis()