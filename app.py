"""
Streamlitアプリケーション

LangGraphを使用したGeminiとChatGPTを統合したチャットアプリ
"""
import streamlit as st
import os
from typing import List, Dict, Any, Optional

from graph.builder import build_graph, run_graph
from utils.helpers import format_messages_for_display, save_conversation_history, load_conversation_history
from config import APP_TITLE, APP_DESCRIPTION, OPENAI_API_KEY, GOOGLE_API_KEY


def initialize_session_state():
    """セッション状態を初期化"""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "current_model" not in st.session_state:
        st.session_state.current_model = "chatgpt"
    if "system_message" not in st.session_state:
        st.session_state.system_message = ""
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "graph" not in st.session_state:
        st.session_state.graph = build_graph()


def display_chat_history():
    """チャット履歴を表示"""
    for message in st.session_state.chat_history:
        with st.chat_message("user" if message["is_user"] else "assistant"):
            st.markdown(message["content"])


def process_user_message(user_input: str):
    """
    ユーザーメッセージを処理し、応答を生成
    
    引数:
        user_input (str): ユーザー入力
    """
    if not user_input.strip():
        return
    
    # ユーザーメッセージを表示
    with st.chat_message("user"):
        st.markdown(user_input)
    
    # チャット履歴に追加
    st.session_state.chat_history.append({"is_user": True, "content": user_input})
    
    # APIキーが設定されているか確認
    if not check_api_keys():
        return
    
    # 応答を生成するための状態を表示
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.markdown("🤔 考え中...")
        
        try:
            # グラフを実行
            result = run_graph(
                st.session_state.graph,
                user_input,
                st.session_state.messages,
                st.session_state.system_message,
                st.session_state.current_model,
            )
            
            # 結果を保存
            st.session_state.messages = result["messages"]
            st.session_state.current_model = result["current_model"]
            response = result["response"]
            
            # アシスタントメッセージを表示
            message_placeholder.markdown(response)
            
            # チャット履歴に追加
            st.session_state.chat_history.append({"is_user": False, "content": response})
            
        except Exception as e:
            error_message = f"エラーが発生しました: {str(e)}"
            message_placeholder.markdown(f"❌ {error_message}")
            st.error(error_message)


def check_api_keys() -> bool:
    """
    APIキーが設定されているか確認
    
    戻り値:
        bool: すべてのAPIキーが設定されている場合はTrue
    """
    missing_keys = []
    
    if not OPENAI_API_KEY:
        missing_keys.append("OpenAI API Key")
    
    if not GOOGLE_API_KEY:
        missing_keys.append("Google API Key")
    
    if missing_keys:
        st.error(f"以下のAPIキーが設定されていません: {', '.join(missing_keys)}")
        st.info("設定方法は以下の通りです：")
        st.code("""
# .envファイルを作成する場合
OPENAI_API_KEY=your_openai_api_key
GOOGLE_API_KEY=your_google_api_key

# または環境変数を設定する場合
export OPENAI_API_KEY=your_openai_api_key
export GOOGLE_API_KEY=your_google_api_key
        """)
        return False
    
    return True


def main():
    """メイン関数"""
    # アプリのタイトルと説明を設定
    st.set_page_config(
        page_title=APP_TITLE,
        page_icon="🤖",
        layout="wide",
    )
    
    # セッション状態を初期化
    initialize_session_state()
    
    # サイドバー
    with st.sidebar:
        st.title("設定")
        
        # システムメッセージ
        system_message = st.text_area(
            "システムメッセージ",
            value=st.session_state.system_message,
            height=150,
            help="両方のモデルに送信されるシステムメッセージを入力してください。",
        )
        
        if system_message != st.session_state.system_message:
            st.session_state.system_message = system_message
        
        # 使用するモデルを選択
        model_choice = st.radio(
            "次回の応答に使用するモデル",
            ["ChatGPT", "Gemini"],
            index=0 if st.session_state.current_model == "chatgpt" else 1,
        )
        
        if (model_choice == "ChatGPT" and st.session_state.current_model != "chatgpt") or \
           (model_choice == "Gemini" and st.session_state.current_model != "gemini"):
            st.session_state.current_model = "chatgpt" if model_choice == "ChatGPT" else "gemini"
        
        st.divider()
        
        # 会話のリセット
        if st.button("会話をリセット", use_container_width=True):
            st.session_state.messages = []
            st.session_state.chat_history = []
            st.rerun()
    
    # メインエリア
    st.title(APP_TITLE)
    st.markdown(APP_DESCRIPTION)
    
    # 現在のモデル情報を表示
    st.info(f"次の応答には **{model_choice}** が使用されます。")
    
    # チャット履歴を表示
    display_chat_history()
    
    # ユーザー入力
    user_input = st.chat_input("メッセージを入力...")
    if user_input:
        process_user_message(user_input)


if __name__ == "__main__":
    main()
