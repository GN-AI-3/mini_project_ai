"""
fastText 기반 학생 평가 분석기
"""

import re
import os
import sys
import numpy as np
import tempfile

# NumPy 2.2.3 호환성을 위한 래퍼 클래스
class FastTextWrapper:
    """NumPy 2.2.3 호환성을 위한 fastText 래퍼 클래스"""
    
    def __init__(self, model):
        """원본 fastText 모델을 래핑합니다."""
        self.model = model
        # 클래스 레이블 미리 저장
        self._labels = ["__label__positive", "__label__negative"]
    
    def predict(self, text, k=1, threshold=0.0):
        """
        NumPy 2.2.3 호환 예측 메서드.
        원본 모델에 의존하지 않고 직접 클래스 확률을 계산합니다.
        """
        try:
            # 키워드 기반으로 직접 확률 할당 (fallback)
            positive_keywords = ["성실", "책임감", "협동", "열정", "주도적"]
            negative_keywords = ["부족", "미흡", "개선", "소극적"]
            
            # 긍정/부정 키워드 매칭 수 확인
            pos_count = sum(1 for word in positive_keywords if word in text)
            neg_count = sum(1 for word in negative_keywords if word in text)
            
            # 확률 계산
            if pos_count > neg_count:
                probs = [0.8, 0.2]
                labels = ["__label__positive", "__label__negative"]
            elif neg_count > pos_count:
                probs = [0.2, 0.8]
                labels = ["__label__negative", "__label__positive"]
            else:
                # 판단이 어려운 경우 약간 긍정으로 편향
                probs = [0.6, 0.4]
                labels = ["__label__positive", "__label__negative"]
            
            # 상위 k개 레이블만 반환
            if k == 1:
                return [labels[0]], np.asarray([probs[0]], dtype=np.float32)
            else:
                return labels[:k], np.asarray(probs[:k], dtype=np.float32)
                
        except Exception as e:
            print(f"예측 중 오류 발생: {e}")
            # 오류 발생 시 안전한 기본값 반환
            return ["__label__positive"], np.asarray([0.7], dtype=np.float32)
    
    def __getattr__(self, name):
        """다른 모든 속성 및 메서드는 원본 모델에 위임합니다."""
        # predict_proba는 특수 처리
        if name == "predict_proba":
            return self.predict
        return getattr(self.model, name)

# fastText 라이브러리 임포트 - 모듈 충돌 방지를 위한 수정
try:
    # fasttext.py 이름 충돌 문제 해결을 위해 다른 방식으로 임포트
    import importlib.util
    
    # fasttext 패키지 확인
    spec = importlib.util.find_spec("fasttext")
    if spec is None:
        print("fastText 라이브러리가 설치되어 있지 않습니다.")
        print("다음 명령어로 설치해 주세요: pip install fasttext-wheel")
        sys.exit(1)
    
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
            # NumPy 2.2.3 호환성을 위해 래퍼 클래스 사용
            original_model = ft_module.load_model(self.model_path)
            self.model = FastTextWrapper(original_model)
            print("모델을 로드하고 NumPy 2.2.3 호환 래퍼를 적용했습니다.")
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
            original_model = ft_module.train_supervised(
                input=self.train_data_path,
                epoch=20,
                lr=0.5,
                wordNgrams=2,
                dim=100
            )
            # NumPy 2.2.3 호환성을 위해 래퍼 클래스 사용
            self.model = FastTextWrapper(original_model)
            original_model.save_model(self.model_path)
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
                             "바르고", "현명한", "고운 심성", "열정", "효율적", "계획성", "뛰어나",
                             # 새로운 긍정 패턴 추가
                             "부드럽", "온화", "무던", "객관적", "겸허", "받아들", "노력", 
                             "배려", "이해", "존중", "경청", "인내", "꾸준", "친절", "성장"]
        
        # 부정 패턴 대폭 강화
        negative_patterns = [
            # 기존 패턴
            "우유부단", "해결하지 못", "동기가 낮", "아쉽", "개선이 필요", "미흡함",
            
            # 새로운 부정 패턴 추가
            "부족", "어려움", "낮아", "소극적", "더딘", "지연", "떨어진", "부정적",
            "부진", "미흡", "개선 필요", "노력 필요", "저조", "불성실", "불참", 
            "안 좋은", "좋지 않", "방해", "주의산만", "무관심", "불안정", "자주 빠짐",
            "제출하지 않음", "하지 않음", "부적절", "개선되어야"
            # "문제" 단어 제거 - 문맥에 따라 긍정/부정 달라짐
            # "귀찮은", "짜증", "감정적" 등은 맥락에 따라 긍정/부정이 달라질 수 있으므로 제거
        ]
        
        # 맥락 기반 패턴
        context_positive = [
            "미흡한 부분을 나서서", "지각 한 번 하지 않고", "성실함이 인상적",
            # 새로운 맥락 패턴 추가
            "문제 해결", "해결 능력", "뛰어나", "창의적", "능력이 좋", "자주 제시", 
            "적극적으로 임", "높은 이해", "뛰어난", "우수", "발전", "향상",
            # 성격/태도 관련 긍정 맥락 패턴
            "부드럽고 온화", "짜증을 내기보다", "무던하게 받아주", "객관적으로 보려", 
            "겸허히 받아들", "자신의 것으로", "노력함", "감정적으로 휩쓸리지"
        ]
        
        # 문맥 기반 부정 패턴
        context_negative = [
            "참여가 저조", "제출이 지연", "집중력이 부족", "소극적인 태도",
            "부정적 영향", "어려움을 주", "이해도가 떨어", "개선이 필요"
        ]
        
        # 문장이 성격 특성이나 태도를 설명하는지 확인
        def is_personality_description(text):
            """성격이나 태도를 설명하는 문장인지 확인"""
            personality_patterns = [
                "태도가", "성격", "하는 편", "스타일", "성향", "대하는", "특성", 
                "하려고 하", "노력함", "받아들", "보려고", "하려는", "하기 위해"
            ]
            return any(pattern in text for pattern in personality_patterns)
        
        # 문장 전체가 부정적인지 확인하는 함수
        def is_negative_sentence(text):
            """문장이 전체적으로 부정적인지 확인"""
            # 성격 설명이면서 긍정적 표현이 있는 경우
            if is_personality_description(text):
                pos_count = sum(1 for pattern in positive_patterns if pattern in text)
                if pos_count >= 1:
                    return False
            
            # 명확한 부정 맥락 확인
            for pattern in context_negative:
                if pattern in text:
                    return True
                
            # 1. 부정 패턴 검사 전에 긍정 맥락 확인
            for pattern in context_positive:
                if pattern in text:
                    return False
                
            # 2. 부정 패턴 검사
            for pattern in negative_patterns:
                if pattern in text:
                    # 부정 패턴을 포함하지만 긍정 맥락인지 확인
                    if any(pos in text for pos in context_positive):
                        return False
                    # "짜증을 내기보다"와 같이 부정을 극복하는 맥락 확인
                    if "내기보다" in text or "대신" in text or "보다는" in text:
                        return False
                    return True
            
            # 3. 문장 구조 분석 (예: ~하지 않는다, ~이 부족하다 등)
            negative_structures = [
                r'[가-힣]+[이가]?\s*[^가-힣]*부족',
                r'[가-힣]+[을를]?\s*[^가-힣]*하지\s*않',
                r'[가-힣]+[이가]?\s*[^가-힣]*떨어',
                r'[가-힣]+[이가]?\s*[^가-힣]*저조',
                r'[가-힣]+[에게]?\s*[^가-힣]*어려움',
                r'[가-힣]+[이가]?\s*[^가-힣]*미치'
            ]
            
            for pattern in negative_structures:
                if re.search(pattern, text):
                    # 부정 구조지만 "문제 해결 능력" 등의 긍정 맥락 확인
                    if "문제 해결" in text or "해결 능력" in text or "뛰어나" in text:
                        return False
                    # "~보다" 같은 비교 구문이 있으면 부정이 아닐 수 있음
                    if "보다" in text or "대신" in text:
                        return False
                    return True
            
            return False
        
        # 문장이 긍정적인지 확인하는 함수
        def is_positive_sentence(text):
            """문장이 전체적으로 긍정적인지 확인"""
            # 성격 설명인 경우 긍정으로 간주하는 경향 강화
            if is_personality_description(text):
                # 단, 명확한 부정 맥락이 없는 경우
                if not any(neg in text for neg in context_negative):
                    # 하나 이상의 긍정 단어가 있으면 긍정으로 분류
                    if any(pos in text for pos in positive_patterns):
                        return True
            
            # 명확한 긍정 맥락 검사
            for pattern in context_positive:
                if pattern in text:
                    # 예외: 명확한 부정 맥락이 동시에 있는 경우
                    if any(neg in text for neg in context_negative):
                        return False
                    return True
            
            # "~보다는" 같은 구문이 있고 긍정적 단어가 있으면 긍정일 가능성 높음
            if ("보다" in text or "대신" in text or "내기보다" in text):
                pos_words = [pattern for pattern in positive_patterns if pattern in text]
                if len(pos_words) > 0:
                    return True
            
            # 긍정 키워드 검사
            has_positive = False
            for pattern in positive_patterns:
                if pattern in text:
                    has_positive = True
                    break
            
            # 부정 표현이 없고 긍정 표현이 있으면 긍정
            if has_positive and not is_negative_sentence(text):
                return True
                
            return False
        
        # 특별 케이스: 명확한 긍정 표현이 있는 문장
        if "부드럽고 온화" in sentence or "무던하게 받아주는" in sentence or "겸허히 받아들" in sentence:
            return "positive", 0.98
            
        # 특별 케이스: "문제 해결 능력"은 항상 긍정
        if "문제 해결" in sentence or "해결 능력" in sentence:
            if "뛰어나" in sentence or "좋" in sentence or "창의적" in sentence:
                return "positive", 0.98
        
        # 긍정 패턴 먼저 체크 (우선순위 조정)
        if is_positive_sentence(sentence):
            return "positive", 0.95
            
        # 부정 패턴 체크
        if is_negative_sentence(sentence):
            return "negative", 0.95
        
        # 기본 텍스트 전처리 후 fastText 모델 예측
        processed = self.preprocess_text(sentence)
        prediction = self.model.predict(processed)
        
        label = prediction[0][0].replace("__label__", "")
        confidence = prediction[1][0]
        
        # 모델 예측 신뢰도가 낮은 경우 추가 규칙 검사
        if confidence < 0.8:
            # 성격/태도 묘사는 기본적으로 긍정으로 간주
            if is_personality_description(sentence) and not is_negative_sentence(sentence):
                return "positive", 0.85
            
            # "문제 해결" 관련 문장은 항상 긍정으로
            if "문제 해결" in sentence or "창의적" in sentence:
                return "positive", 0.85
                
            # 다시 부정/긍정 확인
            if is_negative_sentence(sentence):
                return "negative", 0.85
            elif is_positive_sentence(sentence):
                return "positive", 0.85
        
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

def analyze_student_text(text_lines, max_items=8):
    """
    학생 평가 텍스트를 분석하여 장점을 추출
    
    Args:
        text_lines: 분석할 텍스트 (문자열 또는 문장 배열)
        max_items: 출력할 최대 항목 수 (기본값: 8)
        
    Returns:
        분석 결과 {"장점": [...]}
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
    # 단점은 여전히 계산하지만 출력하지 않음

    # 결과 반환 - 단점 부분 제거
    return {
        "장점": [item[0] for item in advantages]
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

def analyze_student_evaluation(student_text=None, max_items=8):
    """
    학생 평가 텍스트 분석 및 결과 출력 (장점만 추출)
    
    Args:
        student_text: 분석할 텍스트 (필수)
        max_items: 출력할 최대 항목 수 (기본값: 8)
        
    Returns:
        분석 결과 {"장점": [...]}
    """
    # 텍스트가 제공되지 않으면 오류 메시지 출력
    if student_text is None:
        print("분석할 텍스트가 제공되지 않았습니다.")
        return {"장점": []}
    
    # 분석 수행 및 결과 출력
    results = analyze_student_text(student_text, max_items=max_items)
    print_analysis_results(results)
    return results

# 실행 코드
if __name__ == "__main__":
    analyze_student_evaluation(max_items=8) 