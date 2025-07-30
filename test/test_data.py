from transformers import pipeline

# 모델 로드
analyzer = pipeline(
    "sentiment-analysis",
    model="hun3359/klue-bert-base-sentiment",
    top_k=None,
)

# 테스트 텍스트
text = "사랑하는 사람과 있는 멋진 시간"

# 결과 확인
result = analyzer(text)

print("입력:", text)
print("결과 타입:", type(result))
print("결과 길이:", len(result))
print("첫 번째 요소 타입:", type(result[0]))
print("감정 개수:", len(result[0]))
print("\n상위 5개 감정:")
for i, emotion in enumerate(result[0][:5]):
    print(f"{i+1}. {emotion}")

print("\n전체 감정 라벨:")
labels = [item["label"] for item in result[0]]
print(labels)
