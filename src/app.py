__import__("pysqlite3")
import sys

sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")

import streamlit as st
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import ChatMessage
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_upstage import ChatUpstage
from transformers import pipeline
from dotenv import load_dotenv
import os

from utils import StreamHandler, init_conversation, print_conversation


st.set_page_config(page_title="미니 챗봇")
st.title("SOLAR 챗봇 with 감정분석")

# 환경변수 로드
load_dotenv()

try:
    # 배포 환경: Streamlit Cloud secrets 사용
    api_key = st.secrets["UPSTAGE_API_KEY"]
except:
    # 로컬 환경: .env 파일 사용
    api_key = os.getenv("UPSTAGE_API_KEY")


# 파이프라인 초기화
@st.cache_resource
def load_sentiment_analyzer():
    with st.spinner("감정분석 모델 로딩 중..."):
        return pipeline(
            "sentiment-analysis",
            model="hun3359/klue-bert-base-sentiment",
            top_k=None,
        )


# 세션 상태 초기화
init_conversation()


# 사이드바
with st.sidebar:
    session_id = st.text_input("Session ID", value="my_chat_0001")

    # 대화 초기화 버튼
    if st.button("대화 기록 초기화"):
        st.session_state.messages = []
        if session_id in st.session_state.store:
            st.session_state.store[session_id] = ChatMessageHistory()
        st.rerun()


# 세션별 대화 기록 관리
def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in st.session_state.store:
        st.session_state.store[session_id] = ChatMessageHistory()
    return st.session_state.store[session_id]


# 이전 대화 표시
print_conversation()

# 사용자 입력 처리
if user_input := st.chat_input("메시지를 입력하세요..."):
    # 사용자 메시지 표시 및 저장
    with st.chat_message("user"):
        st.write(user_input)
    st.session_state.messages.append(ChatMessage(role="user", content=user_input))

    # 어시스턴트 응답 생성
    with st.chat_message("assistant"):
        with st.spinner("AI가 답변을 생성하고 있습니다..."):
            # 스트리밍 핸들러 설정
            response_container = st.empty()
            stream_handler = StreamHandler(response_container)

            # SOLAR 모델 설정
            llm = ChatUpstage(
                model="solar-mini",
                api_key=api_key,
                streaming=True,
                callbacks=[stream_handler],
            )

            # 기본 프롬프트 설정
            prompt = ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        # "당신은 AI 어시스턴트입니다. 친근하고 정확한 답변을 제공해주세요.",
                        "당신은 불량스러운 성격을 가진 학생입니다. 까칠하고 불친절하게 답변해주세요.",
                    ),
                    MessagesPlaceholder(variable_name="history"),
                    ("human", "{question}"),
                ]
            )

            # 프롬프트와 모델 연결
            runnable = prompt | llm

            # 대화 기록과 함께 실행할 수 있는 체인 생성
            chain_with_memory = RunnableWithMessageHistory(
                runnable,
                get_session_history,
                input_messages_key="question",
                history_messages_key="history",
            )

            # 응답 생성
            try:
                response = chain_with_memory.invoke(
                    {"question": user_input},
                    config={"configurable": {"session_id": session_id}},
                )

                # 스트리밍이 완료된 후 최종 텍스트 저장 (감정 분석용)
                response_text = stream_handler.text

            except Exception as e:
                st.error(f"오류가 발생했습니다: {str(e)}")
                response_text = "죄송합니다. 응답을 생성하는 중 오류가 발생했습니다."
                response_container.write(response_text)

        with st.spinner("감정을 분석하고 있습니다..."):
            # 감정분석 수행
            sentiment_analyzer = load_sentiment_analyzer()
            sentiment_results = sentiment_analyzer(response_text)

            print(sentiment_results)

            # 감정분석 결과 표시
            st.markdown("---")
            st.markdown("**🎭 감정분석 결과:**")

            # 상위 3개 감정만 표시
            top_sentiments = sorted(
                sentiment_results[0], key=lambda x: x["score"], reverse=True
            )[:3]

            for i, result in enumerate(top_sentiments):
                label = result["label"]
                score = result["score"]

                # 순위별 이모지
                rank_emoji = ["🥇", "🥈", "🥉"][i]

                # 프로그레스 바로 시각화
                st.progress(score, text=f"{rank_emoji} {label}: {score:.3f}")

            # 가장 높은 점수의 감정 표시
            best_sentiment = top_sentiments[0]
            st.success(
                f"**주요 감정: {best_sentiment['label']} ({best_sentiment['score']:.3f})**"
            )

        # 어시스턴트 메시지 저장
        st.session_state.messages.append(
            ChatMessage(role="assistant", content=response_text)
        )

        # print("==== 현재 저장된 메시지들: ====")
        # for i, msg in enumerate(st.session_state.messages):
        #     print(f"{i+1}. [{msg.role}] {msg.content}")
