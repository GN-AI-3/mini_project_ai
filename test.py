import os
import warnings

from transformers import pipeline, AutoTokenizer

model = "KoJLabs/bart-speech-style-converter"
tokenizer = AutoTokenizer.from_pretrained(model)

nlg_pipeline = pipeline('text2text-generation', model=model, tokenizer=tokenizer)
# styles = ["문어체", "구어체", "안드로이드", "아재", "채팅", "초등학생", "이모티콘", "enfp", "신사", "할아버지", "할머니", "중학생", "왕", "나루토", "선비", "소심한", "번역기"]
styles = ["구어체"]

for style in styles:
    text = f"{style} 형식으로 변환: 일기 표현력 우수하고 수학 연산능력 뛰어나며 미술에 재주가 엿 보임"
    out = nlg_pipeline(text, max_length=200)
    print(out[0]['generated_text'])