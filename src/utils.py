from langchain_core.callbacks.base import BaseCallbackHandler
import streamlit as st


class StreamHandler(BaseCallbackHandler):
    """스트리임 핸들러"""

    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token: str, **kwargs):
        """새로운 토큰이 생성될 때마다 화면 업데이트"""
        self.text += token
        self.container.markdown(self.text)


def init_conversation():
    """세션 상태 초기화"""
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "store" not in st.session_state:
        st.session_state.store = {}


def print_conversation():
    """이전 대화 기록 표시"""
    for message in st.session_state.messages:
        with st.chat_message(message.role):
            st.write(message.content)
