from transformers import pipeline

models = [
    "hun3359/klue-bert-base-sentiment",
    "alsgyu/sentiment-analysis-fine-tuned-model",
    "monologg/koelectra-base-finetuned-nsmc",
]

test_text = "오늘 날씨가 꽤나 맑아요."

print(f"테스트 문장: {test_text}")
print("=" * 50)

for model_name in models:
    print(f"\n모델: {model_name}")
    try:
        analyzer = pipeline("sentiment-analysis", model=model_name)
        result = analyzer(test_text)
        print(f"결과: {result}")
    except Exception as e:
        print(f"오류: {e}")
    print("-" * 30)
