# 학생 평가 분석기 (Student Evaluation Analyzer)

FastText 기반 학생 평가 텍스트 분석 도구로, 평가 문장을 분석하여 장점(강점)과 단점(개선점)으로 분류해 주는 도구입니다.

## 필요 라이브러리

이 프로그램을 실행하기 위해 다음 라이브러리가 필요합니다:

### 기본 라이브러리

- `re`: 정규 표현식 (Python 기본 제공)
- `os`: 운영체제 인터페이스 (Python 기본 제공)
- `sys`: 시스템 특정 파라미터와 함수 (Python 기본 제공)
- `numpy`: 수치 계산 라이브러리

### 외부 라이브러리

- `fasttext`: 텍스트 분류 모델 라이브러리

## 설치 방법

### 1. 기본 라이브러리 설치

```bash
pip install numpy
```

### 2. fastText 설치

```bash
pip install fasttext-wheel
```

## 사용 방법

### 기본 실행

```python
import student_evaluation as se

# 기본 예제 텍스트로 분석 실행 (상위 5개 장/단점 출력)
se.analyze_student_evaluation(max_items=5)
```

### 사용자 지정 텍스트 분석

```python
import student_evaluation as se

# 분석할 학생 평가 텍스트 준비
text = [
    "학생은 성실하고 책임감 있는 태도로 수업에 참여합니다.",
    "과제 제출이 항상 정확하고 시간을 잘 지킵니다.",
    "다만 토론 참여에 있어서는 좀 더 적극적인 태도가 필요합니다."
]

# 텍스트 분석 및 결과 출력
results = se.analyze_student_text(text, max_items=3)
se.print_analysis_results(results)

# 또는 한번에 분석 및 출력
se.analyze_student_evaluation(text, max_items=3)
```

### 한글 함수명 사용 (선택 사항)

```python
import student_evaluation as se

# 한글 함수명으로도 호출 가능
se.학생_평가_분석(max_items=3)
```

## 텍스트 처리 방식

이 분석기는 다음과 같은 방식으로 텍스트를 처리합니다:

1. 문장 분리: 구두점(., !, ?)과 문맥을 기반으로 텍스트를 의미 있는 문장 단위로 분리
2. 문장 분류: 키워드 기반 분류와 fastText 모델을 사용하여 문장을 긍정(장점)/부정(단점)으로 분류
3. 결과 정렬: 신뢰도 점수에 따라 결과를 정렬하여 가장 유력한 장점과 단점을 상위에 표시

## 참고사항

- 프로그램 첫 실행 시 fastText 모델이 학습되므로 시간이 소요될 수 있습니다.
- 텍스트 전처리는 간단한 공백 기반 토큰화 방식을 사용합니다.
- 학습 모델은 `./data` 디렉토리에 저장됩니다.
