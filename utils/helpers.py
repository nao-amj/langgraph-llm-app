"""
ヘルパー関数モジュール
"""
from typing import Dict, List, Any, Optional
import json


def format_messages_for_display(messages: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    """
    メッセージを表示用にフォーマット

    引数:
        messages (List[Dict[str, str]]): メッセージのリスト

    戻り値:
        List[Dict[str, Any]]: フォーマットされたメッセージのリスト
    """
    formatted_messages = []
    
    for message in messages:
        role = message.get("role", "")
        content = message.get("content", "")
        
        if role == "user":
            formatted_messages.append({"is_user": True, "content": content})
        elif role == "assistant":
            formatted_messages.append({"is_user": False, "content": content})
    
    return formatted_messages


def determine_next_model(state: Dict[str, Any]) -> str:
    """
    次に使用するモデルを決定する

    引数:
        state (Dict[str, Any]): 現在の状態

    戻り値:
        str: 次に使用するモデルの名前 ("chatgpt" または "gemini")
    """
    current_model = state.get("current_model", "")
    
    # シンプルな交互使用ロジック
    if current_model == "chatgpt":
        return "gemini"
    else:
        return "chatgpt"


def save_conversation_history(messages: List[Dict[str, str]], filename: str) -> None:
    """
    会話履歴をJSONファイルに保存

    引数:
        messages (List[Dict[str, str]]): 保存するメッセージのリスト
        filename (str): 保存先のファイル名
    """
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(messages, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"会話履歴の保存中にエラーが発生しました: {e}")


def load_conversation_history(filename: str) -> Optional[List[Dict[str, str]]]:
    """
    JSONファイルから会話履歴を読み込み

    引数:
        filename (str): 読み込むファイル名

    戻り値:
        Optional[List[Dict[str, str]]]: 読み込まれたメッセージのリスト。
        エラーが発生した場合はNone。
    """
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"ファイルが見つかりません: {filename}")
        return None
    except json.JSONDecodeError:
        print(f"JSONの解析エラー: {filename}")
        return None
    except Exception as e:
        print(f"会話履歴の読み込み中にエラーが発生しました: {e}")
        return None
