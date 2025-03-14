from sentence_transformers import SentenceTransformer, util
import numpy as np
from csv_utils import read_from_csv
import re
import pickle
from pykospacing import Spacing

# 상수
PREDEFINED_OUTPUT_CSV_FILE_PATH = "predefined_titles_and_descriptions.csv"
EMBEDDING_DATA_FILE_PATH = "embeddings.pkl"

# 모델 로드 (전역 변수)
embedding_model_instance = None
spacing_instance = Spacing()

def load_embedding_model():
    global embedding_model_instance
    if embedding_model_instance is None:
        print("임베딩 모델 로드 중...")
        embedding_model_instance = SentenceTransformer('sentence-transformers/paraphrase-multilingual-mpnet-base-v2')
    return embedding_model_instance

def generate_and_save_embeddings(predefined_output_csv_file_path, embedding_file_path, target_column_index=0):
    """
    미리 정해진 CSV 파일에서 데이터를 읽고 지정된 열의 문장을 바탕으로 임베딩을 생성한 뒤,
    원본 데이터와 함께 저장하는 함수.

    Args:
        predefined_output_csv_file_path (str): 미리 정해진 CSV 파일 경로.
        embedding_file_path (str): 임베딩 데이터를 저장할 파일 경로.
        target_column_index (int, optional): CSV 데이터에서 사용할 열의 인덱스. 기본값은 0.

    Returns:
        None
    """
    # CSV 파일 읽기
    csv_rows = read_from_csv(predefined_output_csv_file_path)
    target_sentences = [row[target_column_index] for row in csv_rows]

    # 전역 모델을 가져와 임베딩 생성
    embedding_model = load_embedding_model()
    sentence_embeddings = embedding_model.encode(target_sentences)

    # CSV 데이터와 임베딩 데이터를 함께 저장
    with open(embedding_file_path, 'wb') as output_file:
        pickle.dump((csv_rows, sentence_embeddings), output_file)
    print("CSV 데이터와 임베딩이 성공적으로 저장되었습니다.")

def get_most_similar_sentence(input_sentence):
    """
    주어진 문장과 저장된 임베딩 데이터를 활용해 가장 유사한 문장을 찾는 함수.

    Args:
        input_sentence (str): 유사도를 비교할 입력 문장.

    Returns:
        tuple: 유사도 점수, 가장 유사한 문장의 제목, 설명
               (유사도 점수, 제목, 설명).
    """
    # 임베딩 데이터와 CSV 데이터를 로드
    try:
        with open(EMBEDDING_DATA_FILE_PATH, 'rb') as embedding_file:
            csv_data_rows, embedding_vectors = pickle.load(embedding_file)
    except FileNotFoundError:
        raise ValueError(f"임베딩 파일 {EMBEDDING_DATA_FILE_PATH}을(를) 찾을 수 없습니다.")
    except pickle.UnpicklingError:
        raise ValueError(f"임베딩 파일 {EMBEDDING_DATA_FILE_PATH}의 형식이 올바르지 않습니다.")

    # 입력 문장 전처리 및 임베딩 생성
    embedding_model = load_embedding_model()
    preprocessed_text = re.sub(r'\n|\([^)]*\)|\s', '', input_sentence)
    spaced_text = spacing_instance(preprocessed_text)
    input_sentence_embedding = embedding_model.encode(spaced_text)
    print(spaced_text)

    # 코사인 유사도 계산
    similarity_scores = util.cos_sim(input_sentence_embedding, embedding_vectors)
    most_similar_index = np.argmax(similarity_scores)
    most_similar_score = similarity_scores[0][most_similar_index]

    # 가장 유사한 행 데이터 가져오기
    matched_row = csv_data_rows[most_similar_index]
    matched_title = matched_row[0]
    matched_description = matched_row[2]

    return matched_title, matched_description, most_similar_score.item()

# EMBEDDING_DATA_FILE 생성
generate_and_save_embeddings(PREDEFINED_OUTPUT_CSV_FILE_PATH, EMBEDDING_DATA_FILE_PATH, 2)


# 함수 사용 예시

input_sentence = """
어떤 일이든 똑 부러지고 완벽하게 처리하며 공부를 할 때는 무서울 정도의 집중력을 발휘하는 학생으로 매
사에 긍
정적인 사고로 현상을 파악하며 얼굴에 웃음을 항상 머금고 있고 어른에 대한 예의가 몸에 배어있는 학생임. 학교
교육 과정에 대한 두터운 신뢰감을 바탕으로 어느 누구보다 성실하게 임한 결과 지속적으로 내신 성적이 향
상되어
가고 있을 뿐만 아니라 성취도 또한 매우 높은 성적을 보이고 있음. 본인의 진로관련 능력을 향상시키기 위
해 방학
기간 중 열심히 공부하고 노력하고 있음.
(배려) 학습태도가 바르고 언행이 고우며 명랑한 얼굴로 급우들에게 친철하고 대하는 등 타인을 위한 이해심
과 배
려심을 가지고 있음.
(타인존중) 수업시간 도중 모르는 문제로 어려워하는 친구들에게 친절하게 설명해 주고 항상 웃는 모습으로
친구
들을 대함.
(관계지향성) 매사 진지한 자세와 공손함으로 상대방을 대하며 긍정적인 태도로 학급의 모든 친구들과 잘 어
울리
고 교사의 지도와 조언을 경청하고 수용하려는 자세가 돋보임.
학업에 있어서는 계획을 세워 자습시간과 수업에 임하는 자기주도적 학습태도가 잘 갖추어져 학업성적이 전
반적으
로 우수함. 수업시간 집중력이 뛰어나고 집중적인 시간 관리를 위해 스스로 노력하고 있어 더 발전할 것이라
 기대
되는 모범적인 학생임.
(자기주도적 학습능력 영역) 목표를 세우고 열심히 노력하는 모습이 기특한 학생으로 다른 사교육에 의존하
지 않
고 매 교과시간마다 집중하여 공부하는 모습이 한결 같음.
(타인존중) 쉬는 시간과 수업시간 도중 모르는 문제 때문에 어려워하는 친구들에게 친절하게 설명해주고 항
상 웃
는 모습으로 친구들을 대함.
친구 및 학급 내 상황을 빠르게 파악하는 장점을 가지고 있으며, 자신의 감정과 행동을 끊임없이 점검하며,
진지하
고 준비성이 있어 자신의 목표를 위해 최선을 다하는 학생임. 재치 있는 말과 행동으로 주변을 즐겁게 하여
친구들
의 호감을 받는 등 긍정적인 생활태도를 보임.
"""

title, description, similarity = get_most_similar_sentence(input_sentence)

print("가장 유사한 제목:", title)
print("가장 유사한 설명:", description)
print("유사도 점수", similarity)


# # 여러개 테스트
# inputs = read_from_csv("input.csv")
# input_sentences = [row[0] for row in inputs]

# for index, input_sentence in enumerate(input_sentences):
#     print(f"\n{index+1}.")
#     title, description, similarity = get_most_similar_sentence(input_sentence)

#     print("가장 유사한 제목:", title)
#     print("가장 유사한 설명:", description)
#     print("유사도 점수", similarity)

