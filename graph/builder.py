"""
LangGraphビルダー

グラフの構築と実行を担当
"""
from typing import Dict, Any, List, Optional
from langgraph.graph import StateGraph, END

from graph.nodes import (
    GraphState,
    process_user_input,
    router,
    generate_with_chatgpt,
    generate_with_gemini,
)


def build_graph() -> StateGraph:
    """
    LangGraphのワークフローを構築する

    戻り値:
        StateGraph: 構築されたグラフ
    """
    # グラフを初期化
    graph = StateGraph(GraphState)
    
    # ノードを追加
    graph.add_node("process_input", process_user_input)
    graph.add_node("router", router)
    graph.add_node("generate_chatgpt", generate_with_chatgpt)
    graph.add_node("generate_gemini", generate_with_gemini)
    
    # エッジを追加（ノード間の接続）
    graph.add_edge("process_input", "router")
    graph.add_conditional_edges(
        "router",
        {
            "to_chatgpt": "generate_chatgpt",
            "to_gemini": "generate_gemini",
        },
    )
    
    # 各生成ノードから終了状態へ接続
    graph.add_edge("generate_chatgpt", END)
    graph.add_edge("generate_gemini", END)
    
    # 開始ノードを設定
    graph.set_entry_point("process_input")
    
    # コンパイルしたグラフを返す
    return graph.compile()


def run_graph(
    graph: StateGraph,
    user_input: str,
    messages: Optional[List[Dict[str, str]]] = None,
    system_message: str = "",
    current_model: str = "chatgpt",
) -> Dict[str, Any]:
    """
    グラフを実行する

    引数:
        graph (StateGraph): 実行するグラフ
        user_input (str): ユーザー入力
        messages (Optional[List[Dict[str, str]]]): 既存のメッセージ履歴
        system_message (str): システムメッセージ
        current_model (str): 現在のモデル

    戻り値:
        Dict[str, Any]: グラフの実行結果
    """
    # 初期状態を設定
    initial_state = {
        "messages": messages or [],
        "user_input": user_input,
        "system_message": system_message,
        "current_model": current_model,
        "response": "",
    }
    
    # グラフを実行
    result = graph.invoke(initial_state)
    
    return result
