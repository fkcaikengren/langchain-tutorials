from app.config import settings
from dataclasses import dataclass
from typing import Literal
from langchain.agents import create_agent
from langchain.tools import tool, ToolRuntime
from langchain_openai import ChatOpenAI
from langchain.agents.structured_output import ToolStrategy
from langgraph.checkpoint.memory import InMemorySaver


'''
标准的agent创建和使用流程： 
1.定义提示词
2.定义工具
3.构建chat model
4.结构化输出
5.memory
6.创建agent
'''

# 1.定义提示词
SYSTEM_PROMPT = """你是天气助手,如果被问到天气问题，请先确定地点，然后调用相关工具获取实际天气"""


# 2.定义工具

@dataclass
class Context:
    """Custom runtime context schema."""
    user_id: str

@tool
def get_weather_for_location(city: str) -> str:
    """获取指定城市的天气"""
    return f"{city}总是晴日"

@tool
def get_user_location(runtime: ToolRuntime[Context]) -> str:
    """根据用户ID获取用户位置"""
    user_id = runtime.context.user_id
    return "北京" if user_id == "1" else "上海"

# 3.构建chat model
# 符合openai规范的api,可以使用langchain_openai。我们使用 SiliconFlow 提供的 GLM-4.7 模型
# 注意：尽量使用一些新模型，一些旧模型可能会存在一些特性不支持
model = ChatOpenAI(
    model=settings.glm_model,
    base_url=settings.siliconflow_base_url,
    api_key=settings.siliconflow_api_key,
    temperature=0.9,
    max_tokens=5000,
    timeout=60,
)

# 4.结构化输出
# dataclass 和 Pydantic 都是支持的，用来定义结构化输出的格式。
@dataclass
class ResponseFormat:
    """agent的响应格式"""
    # 一语双关的回答 (必要)
    punny_response: str
    # 和天气相关的信息点（可选）
    weather_conditions: str | None = None
    # 字符串，用于描述响应的长度，取值为"short"、"medium"、"long"之一
    length: Literal["short", "medium", "long"] = "short"

# 5.memory
checkpointer = InMemorySaver()


# 6.创建agent
agent = create_agent(
    model,
    system_prompt=SYSTEM_PROMPT,
    tools=[get_user_location, get_weather_for_location],
    context_schema=Context,
    response_format=ToolStrategy(ResponseFormat),
    checkpointer=checkpointer,
    # debug=True, # 开启debug模式，会打印出agent的运行过程
)

# `thread_id`一次会话的唯一标识符
config = {"configurable": {"thread_id": "1"}}

response = agent.invoke(
    {"messages": [{"role": "user", "content": "今天天气如何？"}]},
    config=config,
    context=Context(user_id="1")
)

print(response['structured_response'])
# ResponseFormat(punny_response='今天的天气晴朗温暖，阳光明媚，绝对是一个出去走走的好日子！☀️', weather_conditions='晴日', length='medium')

# 注意：我们可以用同一个`thread_id`继续这个对话.
response = agent.invoke(
    {"messages": [{"role": "user", "content": "thank you!"}]},
    config=config,
    context=Context(user_id="1")
)

print(response['structured_response'])
# ResponseFormat(punny_response='不用客气！很高兴能帮到你！希望你今天有个晴朗愉快的一天！☀️😊', weather_conditions=None, length='medium')