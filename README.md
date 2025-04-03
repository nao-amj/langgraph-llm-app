# LangGraph LLM アプリケーション

LangGraphを使用して、GeminiやChatGPTなどの大規模言語モデル（LLM）を活用した簡易アプリケーション。このリポジトリでは、複数のLLMを組み合わせたワークフローを構築する方法を示しています。

## 機能

- LangGraphを使用したアプリケーションフレームワーク
- Google Gemini APIとOpenAI ChatGPT APIの統合
- マルチステップの会話フロー管理
- Streamlitによるシンプルなユーザーインターフェース

## インストール方法

```bash
# リポジトリをクローン
git clone https://github.com/nao-amj/langgraph-llm-app.git
cd langgraph-llm-app

# 仮想環境を作成（オプション）
python -m venv venv
source venv/bin/activate  # Linuxの場合
# または
venv\Scripts\activate  # Windowsの場合

# 依存パッケージをインストール
pip install -r requirements.txt
```

## 使用方法

1. 環境変数を設定する：
   ```
   export OPENAI_API_KEY=your_openai_api_key
   export GOOGLE_API_KEY=your_google_api_key
   ```

2. アプリケーションを実行する：
   ```
   streamlit run app.py
   ```

3. ブラウザで `http://localhost:8501` にアクセスしてアプリケーションを使用

## プロジェクト構造

```
langgraph-llm-app/
├── app.py                 # Streamlitアプリケーション
├── graph/
│   ├── __init__.py
│   ├── nodes.py           # グラフのノード（LLMなど）
│   └── builder.py         # LangGraphビルダー
├── models/
│   ├── __init__.py
│   ├── base.py            # 基本モデルクラス
│   ├── gemini.py          # Gemini統合
│   └── chatgpt.py         # ChatGPT統合
├── utils/
│   ├── __init__.py
│   └── helpers.py         # ヘルパー関数
├── config.py              # 設定ファイル
├── requirements.txt       # 依存パッケージリスト
└── README.md
```

## ライセンス

MITライセンス
