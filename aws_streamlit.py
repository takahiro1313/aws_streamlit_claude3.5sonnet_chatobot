import json
import boto3
import streamlit as st
from tavily import TavilyClient
import os
from dotenv import load_dotenv
import time
from datetime import datetime, timedelta

load_dotenv()

#boto3のセッションを明示して使用。keyは.envで管理（斎藤）
session = boto3.Session(
    aws_access_key_id=os.getenv("aws_access_key_id"),#自分のものに変更
    aws_secret_access_key=os.getenv("aws_secret_access_key"), #自分のものに変更
    region_name="ap-southeast-2"  # 必要に応じて
)

# Bedrock モデルの設定
# 注意：リクエスト数の制限を緩和するため、クロスリージョンモデルを使用。クロスリージョンモデルを使わない場合はほぼ確実にエラーが出る（斎藤）
#claude3.5 sonnet v2の場合：apac.anthropic.claude-3-5-sonnet-20241022-v2:0 → （斎藤aws環境）1分間に1リクエストのみ使用可能
#claude 3 Haikuの場合：apac.anthropic.claude-3-haiku-20240307-v1:0 → （斎藤aws環境）1分間に4リクエストのみ使用可能
MODEL_ID = "apac.anthropic.claude-3-5-sonnet-20241022-v2:0"

client = session.client("bedrock-runtime", region_name="ap-southeast-2")#リージョンは自分のものに変更

# ==============================
# ツール関連の実装
# ==============================


def web_search(query: str) -> dict:
    """
    Tavilyクライアントを使用してWeb検索を実行する関数

    Args:
        query: 検索クエリ文字列

    Returns:
        検索結果を含む辞書
    """
    #TavilyClient内にapi_keyを明示して使用。
    tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
    return tavily_client.search(query)


# 利用可能なツールのマッピング
tools = {"web_search": web_search}

# ツール設定（ClaudeでWeb_searchを使えるよ、ということを教える→claudeのapiを叩くときに使う）
tool_config = {
    "tools": [
        {
            "toolSpec": {
                "name": "web_search",
                "description": "Web Search",
                "inputSchema": {
                    "json": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Search query",
                            }
                        },
                        "required": ["query"],
                    }
                },
            }
        }
    ]
}


def process_stream(response):
    """
    Bedrockからのストリームレスポンスを処理する関数

    Args:
        response: Bedrockからのレスポンスストリーム

    Returns:
        処理されたメッセージオブジェクト
    """
    content = []
    message = {"content": content}
    text = ""
    tool_use = {}
    reasoning = {}

    # UI出力要素を管理する辞書
    st_out = {}

    for chunk in response["stream"]:
        # メッセージ開始情報の処理
        if "messageStart" in chunk:#roleがuser or assitantを分けて格納
            message["role"] = chunk["messageStart"]["role"]

        # コンテンツブロック開始情報の処理（toolを使う場合の記録）
        elif "contentBlockStart" in chunk:
            tool = chunk["contentBlockStart"]["start"]["toolUse"]
            tool_use["toolUseId"] = tool["toolUseId"]
            tool_use["name"] = tool["name"]

        # コンテンツブロックのデルタ情報処理（chunk毎に送られてくる差分を処理）
        elif "contentBlockDelta" in chunk:
            delta = chunk["contentBlockDelta"]["delta"]
            index = str(chunk["contentBlockDelta"]["contentBlockIndex"])

            # ツール使用情報の処理（ツールを使う場合に検索ワードを繋げる）
            if "toolUse" in delta:
                if "input" not in tool_use:
                    tool_use["input"] = ""
                tool_use["input"] += delta["toolUse"]["input"]

                # UIへのツール使用情報の表示（ツールを使用している時に表示）
                if index not in st_out:
                    st_out[index] = st.expander("Tool use...", expanded=False).empty()
                st_out[index].write(tool_use["input"])

            # テキスト情報の処理
            elif "text" in delta:
                text += delta["text"]

                # UIへのテキスト情報の表示
                if index not in st_out:
                    st_out[index] = st.chat_message("assistant").empty()
                st_out[index].write(text)

            #以下はClaude 3.7のThinkingが使える場合に有効
            # # 推論内容の処理
            # if "reasoningContent" in delta:
            #     if "reasoningText" not in reasoning:
            #         reasoning["reasoningText"] = {"text": "", "signature": ""}

            #     if "text" in delta["reasoningContent"]:
            #         reasoning["reasoningText"]["text"] += delta["reasoningContent"][
            #             "text"
            #         ]
            #     if "signature" in delta["reasoningContent"]:
            #         reasoning["reasoningText"]["signature"] = delta["reasoningContent"][
            #             "signature"
            #         ]

            #     # UIへの推論内容の表示
            #     if index not in st_out:
            #         st_out[index] = st.expander("Thinking...", expanded=True).empty()
            #     st_out[index].write(reasoning["reasoningText"]["text"])

        # コンテンツブロック終了情報の処理
        elif "contentBlockStop" in chunk:
            # ツール使用情報のコンテンツへの追加
            if "input" in tool_use:
                tool_use["input"] = json.loads(tool_use["input"])
                content.append({"toolUse": tool_use})
                tool_use = {}
                
            # 推論内容のコンテンツへの追加
            elif "reasoningText" in reasoning:
                content.append({"reasoningContent": reasoning})
                reasoning = {}
                
            # テキスト情報のコンテンツへの追加
            else:
                content.append({"text": text})
                text = ""

        # メッセージ終了情報の処理
        elif "messageStop" in chunk:
            stop_reason = chunk["messageStop"]["stopReason"]

    return message


# ==============================
# Streamlit UIの実装
# ==============================

st.title("Claude 3.5 Sonnet on Bedrock") #斎藤AWS環境ではClaude 3.7が使えなかったため、3.5 Sonnetを使用
st.subheader("extended thinking with web search")

# セッション状態の初期化（アプリ起動時にメッセージを格納する箱を用意）
if "messages" not in st.session_state:
    st.session_state.messages = []
messages = st.session_state.messages

# 過去のメッセージの表示
for message in messages:
    # テキストコンテンツのみをフィルタリング
    text_content = list(filter(lambda x: "text" in x.keys(), message["content"]))

    for content in text_content:
        with st.chat_message(message["role"]):
            st.write(content["text"])

# ユーザー入力の処理
if prompt := st.chat_input("質問を入力してください"):
    #思考過程を吐かせるプロンプトでラッピング
    wrapped_prompt = f"""以下の問いに、1度だけ思考プロセスを回してから答えてください。
    
    #条件
    1つの質問につき、情報の検索は1度しか行ってはいけない

    Q: {prompt}"""
    with st.chat_message("user"):
        st.write(prompt)

    # ユーザーメッセージの追加
    messages.append({"role": "user", "content": [{"text": wrapped_prompt}]})
    
    # ツール使用のループ処理
    while True:
        # Bedrockモデルへのリクエスト
        response = client.converse_stream(
            modelId=MODEL_ID,
            messages=messages,
            toolConfig=tool_config,
            #claude 3.7以降だと以下のthinking機能も使える（斎藤）
            # additionalModelRequestFields={
            #     "thinking": {
            #         "type": "enabled",
            #         "budget_tokens": 1024,
            #     },
            # },
        )

        # レスポンスの処理とメッセージへの追加
        message = process_stream(response)
        messages.append(message)

        # ツール使用コンテンツのフィルタリング
        tool_use_content = list(
            filter(lambda x: "toolUse" in x.keys(), message["content"])
        )

        # ツール使用がなければループを終了
        if len(tool_use_content) == 0:
            break

        # 各ツール使用の処理
        for content in tool_use_content:
            tool_use_id = content["toolUse"]["toolUseId"]
            name = content["toolUse"]["name"]
            input = content["toolUse"]["input"]

            # ツールの実行と結果の取得
            result = tools[name](**input)

            # ツール結果メッセージの作成
            tool_result = {
                "toolUseId": tool_use_id,
                "content": [{"text": json.dumps(result, ensure_ascii=False)}],
            }

            # ツール結果メッセージの追加
            tool_result_message = {
                "role": "user",
                "content": [{"toolResult": tool_result}],
            }
            messages.append(tool_result_message)
