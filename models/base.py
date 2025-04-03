"""
ベースモデルクラス

すべてのLLMラッパーの基底クラス
"""
from abc import ABC, abstractmethod
from typing import Dict, List, Any


class BaseLanguageModel(ABC):
    """
    言語モデルの基底抽象クラス
    すべてのLLMラッパーはこのクラスを継承する必要があります
    """

    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """
        テキスト生成のための抽象メソッド

        引数:
            prompt (str): モデルへの入力プロンプト
            **kwargs: モデル固有の追加パラメータ

        戻り値:
            str: 生成されたテキスト
        """
        pass

    @abstractmethod
    def generate_with_chat_history(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """
        チャット履歴を考慮したテキスト生成のための抽象メソッド

        引数:
            messages (List[Dict[str, str]]): チャットメッセージのリスト
                各メッセージはrole（"user"または"assistant"）とcontent（内容）を含む辞書
            **kwargs: モデル固有の追加パラメータ

        戻り値:
            str: 生成されたテキスト
        """
        pass

    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """
        モデル情報を取得するための抽象メソッド

        戻り値:
            Dict[str, Any]: モデル名、プロバイダー、バージョンなどの情報を含む辞書
        """
        pass
