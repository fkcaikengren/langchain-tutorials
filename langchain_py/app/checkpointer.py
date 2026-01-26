from langchain.agents.structured_output import ToolStrategy
from app.config import settings
from dataclasses import dataclass
from typing import Literal, Callable
from langchain.agents import create_agent
from langchain.tools import tool, ToolRuntime
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langchain.agents.middleware import wrap_model_call, wrap_tool_call,  ModelRequest, ModelResponse
from langchain.messages import HumanMessage, ToolMessage
from langchain_core.globals import set_debug
from pydantic import BaseModel, Field

'''

'''



@dataclass
class Context:
    """自定义运行时上下文模式。"""
    user_id: str


qwen3_32b_model = ChatOpenAI(
    model=settings.qwen3_32b_model,
    base_url=settings.siliconflow_base_url,
    api_key=settings.siliconflow_api_key,
    temperature=0.5,  
    max_tokens=5000,
    timeout=60,
)


def test_no_checkpointer():
    """
    测试 1: create_agent 不传递 checkpointer。
    预期：没有之前交互的记忆。
    """
    print("\n" + "="*50)
    print("测试 1: create_agent 不带 checkpointer (应该无记忆)")
    print("="*50)
    
    # 1. 创建不带 checkpointer 的 agent
    agent = create_agent(qwen3_32b_model, checkpointer=None)
    
    # 2. 第一次交互
    print("\n【步骤 1】\n [用户]: 嗨！我叫 Bob。")
    response1 = agent.invoke(
        {"messages": [{"role": "user", "content": "嗨！我叫 Bob。"}]},
        {"configurable": {"thread_id": "1"}} # 没有 checkpointer 时 thread_id 可能被忽略，但为了安全起见还是传递
    )
    print(f"[Agent]: {response1['messages'][-1].content}")
    
    # 3. 第二次交互
    print("\n【步骤 2】\n [用户]: 我叫什么名字？")
    # 注意：没有 checkpointer，如果我们想要记忆，必须手动传递对话历史。
    # 但这里我们模拟一个新的回合而不传递历史，预期 agent 不知道名字，
    # 因为状态没有被保存。
    response2 = agent.invoke(
        {"messages": [{"role": "user", "content": "我叫什么名字？"}]},
        {"configurable": {"thread_id": "1"}}
    )
    print(f"[Agent]: {response2['messages'][-1].content}")


def test_with_checkpointer():
    """
    测试 2: create_agent 使用 checkpointer=InMemorySaver()。
    预期：在同一个 thread_id 内有记忆。
    """
    print("\n" + "="*50)
    print("测试 2: create_agent 带 checkpointer (应该有记忆)")
    print("="*50)
    
    # 1. 创建带 checkpointer 的 agent
    memory = InMemorySaver()
    agent = create_agent(qwen3_32b_model, checkpointer=memory)
    thread_config = {"configurable": {"thread_id": "thread-1"}}
    
    # 2. 第一次交互
    print("\n【步骤 1】\n [用户]: 嗨！我叫 Alice。")
    response1 = agent.invoke(
        {"messages": [{"role": "user", "content": "嗨！我叫 Alice。"}]},
        thread_config
    )
    print(f"[Agent]: {response1['messages'][-1].content}")
    
    # 3. 第二次交互
    print("\n【步骤 2】\n [用户]: 我叫什么名字？")
    response2 = agent.invoke(
        {"messages": [{"role": "user", "content": "我叫什么名字？"}]},
        thread_config
    )
    print(f"[Agent]: {response2['messages'][-1].content}")

  
def test_checkpointer_thread_isolation():
    """
    测试 3: create_agent 使用 checkpointer 和不同的 thread_id。
    预期：记忆通过 thread_id 隔离。
    """
    print("\n" + "="*50)
    print("测试 3: create_agent 线程隔离")
    print("="*50)
    
    memory = InMemorySaver()
    agent = create_agent(qwen3_32b_model, checkpointer=memory)
    
    # 1. 线程 A 交互
    print("\n[线程 A] 用户: 嗨！我叫 Charlie。")
    agent.invoke(
        {"messages": [{"role": "user", "content": "嗨！我叫 Charlie。"}]},
        {"configurable": {"thread_id": "thread-A"}}
    )

    # 2. 线程 B 交互 
    print("\n[线程 B] 用户: 你好！我叫 疯狂踩坑人")
    agent.invoke(
        {"messages": [{"role": "user", "content": "你好！我叫 疯狂踩坑人"}]},
        {"configurable": {"thread_id": "thread-B"}}
    )
    
    
    # 3. 线程 A 交互 (问名字)
    print("\n[线程 A] 用户: 我叫什么名字？")
    response_a = agent.invoke(
        {"messages": [{"role": "user", "content": "我叫什么名字？"}]},
        {"configurable": {"thread_id": "thread-A"}}
    )
    print(f"[线程 A] Agent: {response_a['messages'][-1].content}")


    # 4. 线程 B 交互 (问名字)
    print("\n[线程 B] 用户: 我叫什么名字？")
    response_b = agent.invoke(
        {"messages": [{"role": "user", "content": "我叫什么名字？"}]},
        {"configurable": {"thread_id": "thread-B"}}
    )
    print(f"[线程 B] Agent: {response_b['messages'][-1].content}")
    

def test_checkpoints():

    """
    测试 4: 检查 checkpointer 保存的 checkpoint 是否正确。
    """
    print("\n" + "="*50)
    print("测试 4: 检查 checkpointer 保存的 checkpoint")
    print("="*50)

    memory = InMemorySaver()
    agent = create_agent(qwen3_32b_model, checkpointer=memory)
    thread_config = {"configurable": {"thread_id": "thread-1"}}
    
    print("\n[用户]: 嗨！我叫 疯狂踩坑人。")
    response1 = agent.invoke(
        {"messages": [{"role": "user", "content": "嗨！我叫 疯狂踩坑人。"}]},
        thread_config
    )
    print(f"[Agent]: {response1['messages'][-1].content}")
    
    checkpoints = list(memory.list(thread_config))
    print(len(checkpoints), end="\n")
    for checkpoint_tuple in checkpoints:
        print(checkpoint_tuple.checkpoint, end='\n\n')



if __name__ == "__main__":
    
    
    # test_no_checkpointer()
    # test_with_checkpointer()
    test_checkpointer_thread_isolation()

    # test_checkpoints()
