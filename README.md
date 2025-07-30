# SOLAR 챗봇 with 감정분석

SOLAR 모델과 한국어 감정분석을 결합한 간단한 챗봇입니다.

## 기능
- SOLAR 모델 기반 대화
- 세션별 대화 기록 유지  
- 한국어 감정분석 (hun3359/klue-bert-base-sentiment)
- 실시간 스트리밍 응답

## 실행방법

### 1. 설치
```bash
pip install -r "requirements.txt"
```

### 2. API 키 설정
`.env` 파일 생성:
```
UPSTAGE_API_KEY=your_api_key
```

### 3. 실행
```bash
streamlit run src/app.py
```

## 파일구조
```
src/
├── app.py    # 메인 앱
└── utils.py  # 유틸함수
```

## 배포
Streamlit Cloud에서 배포 시 Secrets에 `UPSTAGE_API_KEY` 추가