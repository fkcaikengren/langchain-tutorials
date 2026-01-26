import sqlite3
from app.config import settings
from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver




SYSTEM_PROMPT = "你是一个人工智能助手"


def _build_models() -> tuple[ChatOpenAI, ChatOpenAI]:
    base_kwargs = dict(
        base_url=settings.siliconflow_base_url,
        api_key=settings.siliconflow_api_key,
        timeout=60,
    )

    agent_model = ChatOpenAI(
        model=settings.qwen3_32b_model,
        temperature=0.7,
        max_tokens=2000,
        **base_kwargs,
    )

    summarizer = ChatOpenAI(
        model=settings.qwen3_32b_model,
        temperature=0.2,
        max_tokens=512,
        **base_kwargs,
    )

    return agent_model, summarizer


def test_summarization_middleware() -> None:
    """
    测试 SummarizationMiddleware 中间件。
    """
    agent_model, summarizer = _build_models()
    checkpointer = InMemorySaver()
    agent = create_agent(
        agent_model,
        system_prompt=SYSTEM_PROMPT,
        middleware=[
            SummarizationMiddleware(
                model=summarizer,
                trigger=("messages", 4),
                keep=("messages", 4),
                summary_prefix="对话摘要：",
                summary_prompt="请将以下对话历史压缩成简短的中文摘要，保留关键信息（事实、偏好、约束、决定、结论）：\n{messages}",
            )
        ],
        checkpointer=checkpointer,
    )

    config: RunnableConfig = {"configurable": {"thread_id": "short-memory-demo"}}

    user_inputs = [
        "我叫小明，住在北京。",
        "请记住我更喜欢用中文回答。",
        "我这周想做一个 LangChain 的学习计划。简短控制在100字",
        "计划要按天拆分，每天不超过1小时。简短控制在100字",
        "顺便提醒我：周三晚上要健身。最后请把所有安排再用要点总结一次。",
    ]

    # 自动进行对话
    for idx, text in enumerate(user_inputs, start=1):
        r = agent.invoke({"messages": [{"role": "user", "content": text}]}, config=config)
        messages = r["messages"]
        ai_message = messages[-1]
        
        print(messages)

        print(f"\n[Turn {idx}] 当前上下文消息数：{len(messages)}（trigger=('messages', 4), keep=('messages', 4)）\n")
        print(f"用户：{text}")
        print(f"助手：{ai_message.content}")
       


def test_sqlite_saver() -> None:
    """
    测试 SQLiteSaver 检查点存储。
    """
    agent_model, summarizer = _build_models()
    conn = sqlite3.connect("checkpoints.sqlite", check_same_thread=False)
    memory = SqliteSaver(conn)

    agent = create_agent(
        agent_model,
        system_prompt=SYSTEM_PROMPT,
        checkpointer=memory,
    )

    config: RunnableConfig = {"configurable": {"thread_id": "test_sqlite_saver"}}
    # text = "你好，我叫“疯狂踩坑人”" # 第一次运行，用这个提示词
    text = "请问我叫什么名字？"  # 然后第二次，用这个提示词
    r = agent.invoke({"messages": [{"role": "user", "content": text}]}, config=config)
    messages = r["messages"]
    print(f"[user] {messages[-2].content}")
    print(f"[assistant] {messages[-1].content}")
    # 我最近老是健忘！(尴尬地挠头) 不过既然是"疯狂踩坑人"，那你...


if __name__ == "__main__":
    # test_summarization_middleware()
    test_sqlite_saver()
