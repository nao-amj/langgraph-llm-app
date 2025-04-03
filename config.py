"""
設定ファイル
環境変数と設定パラメータを管理
"""
import os
from dotenv import load_dotenv

# .envファイルから環境変数をロード（存在する場合）
load_dotenv()

# API キー
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")

# OpenAI モデル設定
OPENAI_MODEL = "gpt-4o"

# Google Gemini モデル設定
GEMINI_MODEL = "gemini-1.5-pro"

# LLM設定
DEFAULT_TEMPERATURE = 0.7
MAX_TOKENS = 1024

# アプリケーション設定
APP_TITLE = "LangGraph LLM アプリケーション"
APP_DESCRIPTION = """
LangGraphを使用してGeminiとChatGPTを組み合わせた簡易アプリケーション
"""

# グラフ設定
GRAPH_STATE_TYPE = {
    "messages": list,
    "current_model": str,
    "user_input": str,
    "system_message": str,
    "response": str,
}
