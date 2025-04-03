"""
LangGraphノードの定義

各ノードは状態を受け取り、更新された状態を返す
"""
from typing import Dict, Any, Annotated, TypedDict, List
import json

from models.chatgpt import ChatGPTModel
from models.gemini import GeminiModel
from utils.helpers import determine_next_model


class GraphState(TypedDict):
    """LangGraphの状態を定義するクラス"""
    messages: List[Dict[str, str]]
    current_model: str
    user_input: str
    system_message: str
    response: str


def process_user_input(
    state: GraphState,
) -> Dict[str, Any]:
    """
    ユーザー入力を処理するノード

    引数:
        state (GraphState): 現在のグラフ状態

    戻り値:
        Dict[str, Any]: 更新された状態
    """
    # 新しいユーザーメッセージを追加
    messages = state.get("messages", [])
    user_input = state.get("user_input", "")
    
    if user_input:
        messages.append({"role": "user", "content": user_input})
    
    # 現在のモデルがまだ設定されていない場合、デフォルトでChatGPTを使用
    current_model = state.get("current_model", "chatgpt")
    
    return {
        "messages": messages,
        "current_model": current_model,
    }


def router(state: GraphState) -> str:
    """
    使用するモデルを決定するルーターノード

    引数:
        state (GraphState): 現在のグラフ状態

    戻り値:
        str: 次のエッジの名前 ("to_chatgpt" または "to_gemini")
    """
    current_model = state.get("current_model", "chatgpt")
    
    if current_model == "chatgpt":
        return "to_chatgpt"
    else:
        return "to_gemini"


def generate_with_chatgpt(
    state: GraphState,
) -> Dict[str, Any]:
    """
    ChatGPTを使用して応答を生成するノード

    引数:
        state (GraphState): 現在のグラフ状態

    戻り値:
        Dict[str, Any]: 更新された状態
    """
    messages = state.get("messages", [])
    system_message = state.get("system_message", "")

    # ChatGPTモデルをインスタンス化
    model = ChatGPTModel()
    
    # 応答を生成
    response = model.generate_with_chat_history(messages, system_message=system_message)
    
    # アシスタントメッセージを追加
    messages.append({"role": "assistant", "content": response})
    
    # 次のモデルを決定
    next_model = determine_next_model(state)
    
    return {
        "messages": messages,
        "response": response,
        "current_model": next_model,
    }


def generate_with_gemini(
    state: GraphState,
) -> Dict[str, Any]:
    """
    Google Geminiを使用して応答を生成するノード

    引数:
        state (GraphState): 現在のグラフ状態

    戻り値:
        Dict[str, Any]: 更新された状態
    """
    messages = state.get("messages", [])
    system_message = state.get("system_message", "")

    # Geminiモデルをインスタンス化
    model = GeminiModel()
    
    # 応答を生成
    response = model.generate_with_chat_history(messages, system_message=system_message)
    
    # アシスタントメッセージを追加
    messages.append({"role": "assistant", "content": response})
    
    # 次のモデルを決定
    next_model = determine_next_model(state)
    
    return {
        "messages": messages,
        "response": response,
        "current_model": next_model,
    }
