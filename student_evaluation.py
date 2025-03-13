"""
fastText 기반 학생 평가 분석기
"""

import re
import os
import sys
import numpy as np
import tempfile

# fastText 라이브러리 임포트 - 모듈 충돌 방지를 위한 수정
try:
    # fasttext.py 이름 충돌 문제 해결을 위해 다른 방식으로 임포트
    import importlib.util
    
    # fasttext 패키지 확인
    spec = importlib.util.find_spec("fasttext")
    if spec is None:
        print("fastText 라이브러리가 설치되어 있지 않습니다. 설치 중...")
        import pip
        pip.main(['install', 'fasttext-wheel'])
    
    # 현재 파일 경로 확인
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # sys.path에서 현재 디렉토리 제거 (파일 이름 충돌 방지)
    if current_dir in sys.path:
        sys.path.remove(current_dir)
    
    # 이제 fasttext 모듈 임포트
    import fasttext as ft_module
except ImportError as e:
    print(f"fastText 라이브러리 임포트 중 오류: {e}")
    print("pip install fasttext-wheel로 설치해 주세요.")
    sys.exit(1)

# ===== 문장 분리 함수 =====
def split_sentences(text):
    """텍스트를 문장으로 분리"""
    if isinstance(text, list):
        text = ' '.join(text)
    
    # 카테고리 패턴 및 문장 끝 패턴 매칭
    category_pattern = r'\(([^)]+)\)[^.!?]*[.!?]?'
    sentence_end_pattern = r'[.!?]\s+(?=[A-Z가-힣])'
    
    # 경계 위치 수집
    boundaries = []
    for match in re.finditer(category_pattern, text):
        boundaries.append(match.start())
    
    for match in re.finditer(sentence_end_pattern, text):
        boundaries.append(match.end() - 1)
    
    boundaries.sort()
    
    # 문장 수집
    sentences = []
    start = 0
    
    for boundary in boundaries:
        if boundary > start:
            sentence = text[start:boundary+1].strip()
            if len(sentence) >= 10:
                sentences.append(sentence)
            start = boundary + 1
    
    # 마지막 문장 추가
    if start < len(text):
        sentence = text[start:].strip()
        if len(sentence) >= 10:
            sentences.append(sentence)
    
    # 문장이 완전하지 않은 경우 처리
    def is_incomplete(s):
        return len(s) < 15 or not re.search(r'[.!?]$', s)
    
    result = []
    for i, s in enumerate(sentences):
        if is_incomplete(s) and i < len(sentences) - 1:
            result.append(s + ' ' + sentences[i+1])
            i += 1
        else:
            result.append(s)
    
    return [s.strip() for s in result if len(s.strip()) >= 10]

# fastText 기반 분석기 구현
class FastTextClassifier:
    def __init__(self):
        # 데이터 디렉토리 및 모델 경로 설정
        self.data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
        os.makedirs(self.data_dir, exist_ok=True)
        self.model_path = os.path.join(self.data_dir, 'fasttext_student_eval.bin')
        self.train_data_path = os.path.join(self.data_dir, 'train_data.txt')
        
        # 예시 문장 데이터 (핵심만 유지)
        self.positive_examples = [
            "성실하고 책임감 있는 태도로 학업에 임함",
            "협동심이 강하고 팀워크에 기여하는 모습을 보임",
            "학업에 열정을 가지고 꾸준한 발전을 이루고 있음",
            "자기주도 학습 능력이 우수하여 스스로 성장함",
            "음악적 재능이 뛰어남", "악기 연주 능력이 우수함"
        ]
        
        self.negative_examples = [
            "수업 참여도가 낮고 학업에 소극적인 모습을 보임",
            "과제 제출 및 학습 태도에 개선이 필요함",
            "동료와의 협력보다는 자기 중심적인 경향을 보임",
            "학습 동기가 부족하여 꾸준한 성장을 이루지 못함",
            "우유부단한 태도로 결정을 내리지 못함"
        ]
        
        # 모델 로드 또는 생성
        if os.path.exists(self.model_path):
            self.model = ft_module.load_model(self.model_path)
        else:
            print("새 모델을 훈련합니다...")
            self.prepare_training_data()
            self.train_model()
    
    def prepare_training_data(self):
        """훈련 데이터 준비"""
        with open(self.train_data_path, 'w', encoding='utf-8') as f:
            for pos_ex in self.positive_examples:
                f.write(f"__label__positive {pos_ex}\n")
            for neg_ex in self.negative_examples:
                f.write(f"__label__negative {neg_ex}\n")
        
        print(f"훈련 데이터 준비 완료: {self.train_data_path}")
    
    def train_model(self):
        """fastText 모델 훈련"""
        try:
            self.model = ft_module.train_supervised(
                input=self.train_data_path,
                epoch=20,
                lr=0.5,
                wordNgrams=2,
                dim=100
            )
            self.model.save_model(self.model_path)
            print(f"모델 훈련 및 저장 완료: {self.model_path}")
        except Exception as e:
            print(f"모델 훈련 중 오류 발생: {e}")
    
    def preprocess_text(self, text):
        """텍스트 전처리 (단순 공백 기반 토큰화)"""
        # 단순한 공백 기반 토큰화로 대체
        return ' '.join(text.split())
    
    def classify_sentence(self, sentence):
        """문장 분류 (긍정/부정)"""
        # 문장이 비어있거나 짧으면 빈 결과 반환
        if not sentence or len(sentence.strip()) < 5:
            return "neutral", 0.0
        
        # 키워드 기반 분류 먼저 시도
        positive_patterns = ["성실", "책임감", "자기주도", "협동", "규칙준수", "나눔", "배려", 
                             "관계", "봉사", "성취", "집중력", "공동체", "모범", "등교", "지각 없", 
                             "청소", "자발적", "깨끗이", "나서서", "앞장", "봉사", "의욕", "적극적", 
                             "바르고", "현명한", "고운 심성", "열정", "효율적", "계획성", "뛰어나"]
        
        negative_patterns = ["우유부단", "해결하지 못", "동기가 낮", "아쉽", "개선이 필요", "미흡함"]
        
        # 맥락 기반 패턴
        context_positive = ["미흡한 부분을 나서서", "지각 한 번 하지 않고", "성실함이 인상적"]
        
        # 긍정 패턴 체크
        for pattern in context_positive:
            if pattern in sentence:
                return "positive", 0.95
            
        for pattern in positive_patterns:
            if pattern in sentence:
                return "positive", 0.9
        
        # 부정 패턴 체크
        for pattern in negative_patterns:
            if pattern in sentence:
                # 부정 키워드가 있지만 맥락이 긍정적인지 확인
                if any(pos in sentence for pos in context_positive):
                    return "positive", 0.95
                return "negative", 0.9
        
        # 기본 텍스트 전처리 후 fastText 모델 예측
        processed = self.preprocess_text(sentence)
        prediction = self.model.predict(processed)
        
        label = prediction[0][0].replace("__label__", "")
        confidence = prediction[1][0]
        
        # 기본값은 긍정으로 설정 (신뢰도가 낮은 경우)
        if label == "negative" and confidence < 0.7:
            return "positive", 0.7
        
        # 분류 결과 반환
        return label, confidence
    
    def analyze_text(self, text):
        """텍스트 분석 (여러 문장 처리)"""
        # 텍스트를 문장으로 분리
        sentences = split_sentences(text)
        
        advantages = []
        disadvantages = []
        
        for sentence in sentences:
            if len(sentence.strip()) < 10:  # 너무 짧은 문장은 건너뛰기
                continue
            
            category, confidence = self.classify_sentence(sentence)
            if category == "positive":
                advantages.append((sentence, confidence))
            else:
                disadvantages.append((sentence, confidence))
        
        # 신뢰도 순으로 정렬
        advantages.sort(key=lambda x: x[1], reverse=True)
        disadvantages.sort(key=lambda x: x[1], reverse=True)
        
        return {
            "장점": advantages,
            "단점": disadvantages
        }

# 전역 인스턴스 생성 (싱글톤 패턴)
_classifier_instance = None

def get_classifier():
    """싱글톤 분류기 인스턴스 반환"""
    global _classifier_instance
    if _classifier_instance is None:
        _classifier_instance = FastTextClassifier()
    return _classifier_instance

# ===== 공개 API 함수 =====

def analyze_student_text(text_lines, max_items=5):
    """
    학생 평가 텍스트를 분석하여 장점과 단점을 추출
    
    Args:
        text_lines: 분석할 텍스트 (문자열 또는 문장 배열)
        max_items: 각 카테고리별 출력할 최대 항목 수 (기본값: 5)
        
    Returns:
        분석 결과 {"장점": [...], "단점": [...]}
    """
    classifier = get_classifier()
    sentences = split_sentences(text_lines)
    
    advantages = []
    disadvantages = []
    
    for sentence in sentences:
        if len(sentence.strip()) < 10:  # 너무 짧은 문장은 건너뛰기
            continue
        
        category, confidence = classifier.classify_sentence(sentence)
        if category == "positive":
            advantages.append((sentence, confidence))
        else:
            disadvantages.append((sentence, confidence))
    
    # 신뢰도 기준 정렬
    advantages.sort(key=lambda x: x[1], reverse=True)
    disadvantages.sort(key=lambda x: x[1], reverse=True)
    
    # 최대 항목 수 제한
    advantages = advantages[:max_items]
    disadvantages = disadvantages[:max_items]
    
    # 결과 반환
    return {
        "장점": [item[0] for item in advantages],
        "단점": [item[0] for item in disadvantages]
    }

def print_analysis_results(results):
    """
    분석 결과 출력
    
    Args:
        results: analyze_student_text 함수의 결과
    """
    print("\n=== 학생 평가 분석 결과 ===")
    
    print("\n[장점]")
    for i, advantage in enumerate(results["장점"], 1):
        # 긴 문장은 80자 기준으로 줄바꿈하여 출력
        print(f"{i}. {advantage}")
        
    print("\n[단점]")
    for i, disadvantage in enumerate(results["단점"], 1):
        print(f"{i}. {disadvantage}")

def analyze_student_evaluation(student_text=None, max_items=5):
    """
    학생 평가 텍스트 분석 및 결과 출력 (장점/단점 추출)
    
    Args:
        student_text: 분석할 텍스트 (기본값: 예제 사용)
        max_items: 각 카테고리별 출력할 최대 항목 수
        
    Returns:
        분석 결과 {"장점": [...], "단점": [...]}
    """
    # 기본 예제
    if student_text is None:
        student_text = [
            "자기 통제력이 강하여 어려운 일에도 마음이 쉽게 흔들리지 않는 장점을 가진 학생으로 매사 성실하며 책임감이 강함. 맡은 역할들도 모두 충실히 임함.",
            "구체적인 진로 희망 관련 활동을 수행하고 진로 목표를 꾸준히 실천하려고 노력하는 모습이 보임.",
            "친구 및 교내 봉사활동 등에도 관심을 두고 지속적으로 참여하며 함께 협력하고 도우며 나누는 활동을 실천함.",
            "또한 교내외 대회에 적극적으로 참여함.",
            "학습 태도와 의욕을 내고 노력에 대해 아쉬운 부분도 있지만, 전반적으로 꾸준히 학습에 임하며 지속적인 성적 관리를 위해 스스로 계획하고 목표 설정에 따른 학습 전략을 실천하려고 노력함.",
            "여러 과목에서 흥미를 보이고 의문점을 적극적으로 해결하려고 함.",
            "2019년도 신산업특성화고 실습 및 프로젝트 경험을 통해 사물인터넷(IoT), 머신 러닝, 3D 인쇄 등의 분야에 흥미를 가지게 되었고, 이를 토대로 대학 전공 분야를 선택하고자 하는 의지가 분명함.",
            "교내 기술경진대회에서 소프트웨어 프로그래밍을 주제로 출전하여 창의적인 아이디어를 발표함.",
            "학생 스스로 기술적인 능력을 발전시키고자 관련 동아리 활동과 방과후 학습에 열심히 참여함.",
            "남다른 탐구심으로 무언가 새롭게 개발하고 배우는 것을 즐기며 이를 토대로 미래의 진로 분야에서 전문가가 되기를 꿈꾸고 있음.",
            "이러한 열정을 통해 자신이 세운 모든 것을 이루어낼 수 있는 실력을 갖출 것으로 기대됨.",
            "예의가 바르고 친구들과도 원만히 지내면서 협력하는 태도를 갖춘 학생으로 다소 내성적이지만 성실하게 학급 활동에 참여하려고 노력함.",
            "수업 중 질문을 적극적으로 하며 자신이 이해하지 못한 부분을 해결하려고 함.",
            "매사에 꾸준한 자세를 보이고 매 순간 최선을 다해 학습에 임함.",
            "특히 정보통신 분야에 대한 높은 관심과 흥미로 관련 책을 스스로 찾아 읽고, 실습 활동을 즐겨 함.",
            "이론과 실습을 결합해 보는 과정을 통해 지식과 기술을 향상시키고자 노력하며, 앞으로도 자신의 꿈을 구체화하기 위한 준비와 탐색을 계속할 것으로 기대됨."
        ]
    
    # 분석 수행 및 결과 출력
    results = analyze_student_text(student_text, max_items=max_items)
    print_analysis_results(results)
    return results

# 한글 함수명 유지 (호환성)
def 학생_평가_분석(학생_평가_텍스트=None, 최대_항목_수=5):
    return analyze_student_evaluation(학생_평가_텍스트, 최대_항목_수)

# 실행 코드
if __name__ == "__main__":
    analyze_student_evaluation(max_items=5) 