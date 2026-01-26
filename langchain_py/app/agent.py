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
中间件
模型（静态和动态）
工具
响应格式（结构化输出）
状态检查（记忆）
'''


set_debug(True)


@dataclass
class Context:
    """Custom runtime context schema."""
    user_id: str

@tool
def compare_two_numbers(a: float, b: float) -> int:
    """
    比较两个数字a，b的大小
    Args:
        a (float): 第一个数字
        b (float): 第二个数字
    Returns:
        int: 比较结果. 如果a>b返回1，a<b返回-1，a=b返回0
    """
    if a > b:
        return 1
    elif a < b:
        return -1
    else:
        return 0

@tool
def get_user_location(runtime: ToolRuntime[Context]) -> str:
    """根据用户ID获取用户位置"""
    user_id = runtime.context.user_id
    return "北京" if user_id == "1" else "上海"



glm_model = ChatOpenAI(
    model=settings.glm_model,
    base_url=settings.siliconflow_base_url,
    api_key=settings.siliconflow_api_key,
    temperature=0.9,
    max_tokens=10000,
    timeout=60,
)

ds_model = ChatOpenAI(
    model=settings.ds_model,
    base_url=settings.siliconflow_base_url,
    api_key=settings.siliconflow_api_key,
    temperature=0.9,
    max_tokens=10000,
    timeout=60,
)

qwen3_32b_model = ChatOpenAI(
    model=settings.qwen3_32b_model,
    base_url=settings.siliconflow_base_url,
    api_key=settings.siliconflow_api_key,
    temperature=0.9,
    max_tokens=5000,
    timeout=60,
)

def _extract_latest_user_text(messages: list) -> str:
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            return str(message.content)
    if not messages:
        return ""
    last = messages[-1]
    content = getattr(last, "content", None)
    if content is not None:
        return str(content)
    return str(last)


def _judge_complexity(user_text: str) -> Literal["simple", "complex"]:
    router = qwen3_32b_model.bind(max_tokens=64) # 控制输出长度，避免冗余信息输出
    res = router.invoke(
        [
            {
                "role": "system",
                "content": "你是问题复杂度分类器。根据用户问题判断复杂度：\n- simple：单一事实/常识问答、简单翻译/润色、很短的直接回答、无需多步推理或设计。\n- complex：需要多步推理、方案设计/架构、长文写作、复杂代码/调试、严谨数学推导、对比权衡。\n只输出：simple 或 complex。",
            },
            {"role": "user", "content": user_text},
        ]
    )
    text = str(getattr(res, "content", res)).strip().lower()
    if text == "simple" or "simple" in text or "简单" in text:
        return "simple"
    if text == "complex" or "complex" in text or "复杂" in text:
        return "complex"
    return "complex"


@wrap_model_call
def dynamic_model_selection(request: ModelRequest, handler: Callable[[ModelRequest], ModelResponse]) -> ModelResponse:
    """Choose model based on conversation complexity."""
    user_text = _extract_latest_user_text(request.messages)
    complexity = _judge_complexity(user_text)
    selected_model = ds_model if complexity == "simple" else glm_model
    return handler(request.override(model=selected_model))


@wrap_tool_call
def handle_tool_errors(request: ModelRequest, handler: Callable[[ModelRequest], ModelResponse]) -> ModelResponse:
    """处理工具调用错误，返回自定义错误消息"""
    try:
        return handler(request)
    except Exception as e:
        # 返回自定义的错误消息给LLM
        return ToolMessage(
            content=f"Tool error: Please check your input and try again. ({str(e)})",
            tool_call_id=request.tool_call["id"]
        )

def test_dynamic_model_selection():
    """
    """
    checkpointer = InMemorySaver()

    agent = create_agent(
        glm_model,
        context_schema=Context,
        checkpointer=checkpointer,
        middleware=[dynamic_model_selection],
    )

    config = {"configurable": {"thread_id": "1"}}

    r1 = agent.invoke(
        {"messages": [{"role": "user", "content": "1.9 和1.11 哪个数字大？(这个问题有点难，请一步步思考)"}]},
        config=config,
        context=Context(user_id="1"),
    )
    # print(r1)
    ai_message = r1["messages"][-1]
    print("响应内容：", end="\n")
    print(ai_message.content)
    print(f"调用模型：\n {ai_message.response_metadata['model_name']}")
    """
    响应内容：
    ...
    所以答案是：**1.9 比 1.11 大**。
    虽然 1.11 的数字位数更多，但在比较小数大小时，我们主要看数值本身。1.9 实际上比 2 只少 0.1，而 1.11 比 2 少 0.89，所以 1.9 明显更大。
    调用模型：
    deepseek-ai/DeepSeek-V3.2-Exp
    """
    r2 = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "请用langchain 1.x 设计一个简单的问答系统，用户可以向系统咨询某地的天气信息，包括天气工具调用。",
                }
            ]
        },
        config=config,
        context=Context(user_id="1"),
    )
    ai_message = r2["messages"][-1]
    print("响应内容：", end="\n")
    print(ai_message.content)
    print(f"调用模型：\n {ai_message.response_metadata['model_name']}")
    """
    响应内容：
    # LangChain 1.x 天气问答系统设计

    下面是一个使用 LangChain 1.x 设计的简单天气问答系统完整示例：
    ...
    调用模型：
    Pro/zai-org/GLM-4.7
    """
    


def test_tool_compare_two_numbers():
    """
    测试 compare_two_numbers 工具. 
    加入 handle_tool_errors 中间件
    """
    checkpointer = InMemorySaver()

    agent = create_agent(
        ds_model,
        context_schema=Context,
        checkpointer=checkpointer,
        tools=[compare_two_numbers],
        middleware=[dynamic_model_selection, handle_tool_errors],
    )
    r = agent.invoke(
        {"messages": [{"role": "user", "content": "1.9 和1.11 哪个数字大？"}]},
        config={"configurable": {"thread_id": "1"}},
        context=Context(user_id="1"),
    )

    ai_message = r["messages"][-1]
    print("响应内容：", end="\n")
    print(ai_message.content)
    print(f"调用模型：\n {ai_message.response_metadata['model_name']}")
    # 响应内容：
    # 根据比较结果，**1.9 比 1.11 大**。
    # 调用模型：
    # deepseek-ai/DeepSeek-V3.2-Exp
    """
    最后一次prompt:
        Human: 1.9 和1.11 哪个数字大？
        AI: 我来帮你比较这两个数字。\n\n[{'name': 'compare_two_numbers', 'args': {'a': 1.9, 'b': 1.11}, 'id': '019bf8eb9ca83173071547a05e3a3fe0', 'type': 'tool_call'}]
        Tool: 1
    """



class CompareResult(BaseModel):
    num1: float  = Field(..., description="第一个数字")
    num2: float  = Field(..., description="第二个数字")
    result: int  = Field(..., description="比较结果，1 表示 num1 大于 num2，-1 表示 num1 小于 num2，0 表示相等")


def test_response_fomat():
    """
    测试响应格式是否符合要求
    """
    checkpointer = InMemorySaver()

    agent = create_agent(
        glm_model,
        context_schema=Context,
        checkpointer=checkpointer,
        tools=[compare_two_numbers],
        middleware=[handle_tool_errors],
        response_format=CompareResult, # 约束格式
    )
    r = agent.invoke(
        {"messages": [{"role": "user", "content": "1.9 和1.11 哪个数字大？"}]},
        config={"configurable": {"thread_id": "1"}},
        context=Context(user_id="1"),
    )
    
    print(r["structured_response"])
    # 输出 CompareResult(num1=1.9 num2=1.11 result=1)
    print(r)
    """r是字典，结构如下：
    {
        messages: [],
        model_name: "deepseek-ai/DeepSeek-V3.2-Exp",
        ...
        # response_format带来的字段（此时messages最后的AIMessage的content是空的）
        structured_response: CompareResult(num1=1.9 num2=1.11 result=1) 
    }
    """

if __name__ == "__main__":
    # test_dynamic_model_selection()
    # test_tool_compare_two_numbers()
    test_response_fomat()
