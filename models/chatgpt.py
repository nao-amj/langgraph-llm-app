"""
ChatGPT (OpenAI) モデルラッパー
"""
from typing import Dict, List, Any, Optional

from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, AIMessage, SystemMessage

from models.base import BaseLanguageModel
from config import OPENAI_API_KEY, OPENAI_MODEL, DEFAULT_TEMPERATURE, MAX_TOKENS


class ChatGPTModel(BaseLanguageModel):
    """OpenAI ChatGPT モデルのラッパークラス"""

    def __init__(
        self,
        model_name: str = OPENAI_MODEL,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int = MAX_TOKENS,
        api_key: Optional[str] = OPENAI_API_KEY,
    ):
        """
        初期化メソッド

        引数:
            model_name (str): 使用するモデル名
            temperature (float): 生成の多様性を制御するパラメータ
            max_tokens (int): 生成するトークンの最大数
            api_key (Optional[str]): OpenAI APIキー
        """
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.api_key = api_key

        # LangChain ChatOpenAIモデルをインスタンス化
        self.model = ChatOpenAI(
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            openai_api_key=api_key,
        )

    def generate(self, prompt: str, system_message: str = None, **kwargs) -> str:
        """
        テキスト生成

        引数:
            prompt (str): モデルへの入力プロンプト
            system_message (str, optional): システムメッセージ
            **kwargs: モデル固有の追加パラメータ

        戻り値:
            str: 生成されたテキスト
        """
        messages = []
        
        # システムメッセージがある場合は追加
        if system_message:
            messages.append(SystemMessage(content=system_message))
        
        # ユーザープロンプトを追加
        messages.append(HumanMessage(content=prompt))
        
        # 応答を生成
        response = self.model.invoke(messages)
        
        return response.content

    def generate_with_chat_history(
        self, messages: List[Dict[str, str]], system_message: str = None, **kwargs
    ) -> str:
        """
        チャット履歴を考慮したテキスト生成

        引数:
            messages (List[Dict[str, str]]): チャットメッセージのリスト
                各メッセージはrole（"user"または"assistant"）とcontent（内容）を含む辞書
            system_message (str, optional): システムメッセージ
            **kwargs: モデル固有の追加パラメータ

        戻り値:
            str: 生成されたテキスト
        """
        langchain_messages = []
        
        # システムメッセージがある場合は追加
        if system_message:
            langchain_messages.append(SystemMessage(content=system_message))
        
        # メッセージ履歴を変換
        for message in messages:
            if message["role"] == "user":
                langchain_messages.append(HumanMessage(content=message["content"]))
            elif message["role"] == "assistant":
                langchain_messages.append(AIMessage(content=message["content"]))
            elif message["role"] == "system":
                langchain_messages.append(SystemMessage(content=message["content"]))
        
        # 応答を生成
        response = self.model.invoke(langchain_messages)
        
        return response.content

    def get_model_info(self) -> Dict[str, Any]:
        """
        モデル情報を取得

        戻り値:
            Dict[str, Any]: モデル情報を含む辞書
        """
        return {
            "name": self.model_name,
            "provider": "OpenAI",
            "type": "ChatGPT",
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
