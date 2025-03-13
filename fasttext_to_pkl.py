import csv
from datetime import datetime
import pickle
from gensim.models.fasttext import load_facebook_model

# 최초 실행에서 모델 로드 및 저장
def load_and_save_model():
    print(f"== LOAD fasttext START at {datetime.now()} ==")
    model = load_facebook_model('model\cc.ko.300.bin')
    with open('fasttext_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    print(f"== LOAD fasttext   END at {datetime.now()} ==")
    return model

# Pickle에서 모델 불러오기
def load_from_pickle():
    print(f"== LOAD fasttext START at {datetime.now()} ==")
    with open('fasttext_model.pkl', 'rb') as f:
        model = pickle.load(f)
    print(f"== LOAD fasttext   END at {datetime.now()} ==")
    return model

# 한 번만 로드
try:
    model = load_from_pickle()
except FileNotFoundError:
    model = load_and_save_model()

results = []
with open('data/output.csv', mode='r', encoding='utf-8') as file:
    reader = csv.reader(file)
    for row in reader:
        keyword = row[1]
        similar_words = model.wv.most_similar(keyword, topn=3)
        results.append(f"{keyword} - {similar_words}")

print(results)