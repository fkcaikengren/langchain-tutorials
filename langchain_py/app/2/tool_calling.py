
import json
from dataclasses import dataclass
from typing import Literal
from app.config import settings
from langchain.agents import create_agent
from langchain_core.globals import set_debug
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langchain.tools import tool, ToolRuntime

set_debug(True)


model = ChatOpenAI(
    model=settings.glm_model,
    base_url=settings.siliconflow_base_url,
    api_key=settings.siliconflow_api_key,
    temperature=0.9,
    max_tokens=5000,
    timeout=60,
)

@tool
def get_reviews(positive: bool) -> list[str]:
    """
    获取罗小黑电影评论列表
    Args:
        positive: 是否获取正面评论, True 表示获取正面评论, False 表示获取负面评论
    Returns:
        评论列表
    """
    positive_reviews = [
        "原来两三岁的小孩也可以不扯女孩裙子啊；原来不整屎尿屁也可以做出让全场大笑的效果啊；原来女角色也可以不穿超短裙高开叉高跟鞋啊；原来男师父女徒弟也可以不暧昧纯师徒情啊；原来一个动画片里正派之间也可以有不同的价值观啊；原来不喊口号不献祭亲朋好友父老乡亲也能表达反战的思想啊。罗小黑你还是太超前了。",
        "瑕不掩瑜。非常好的一点是，一点儿爹味都没有，不judge任何人（妖精），没有任何人（妖精）需要被打败或悔过。这在中国的大型说教重灾区———国漫中已是十分可贵。",
        "“无限虽然爱装逼，但是他没有跟鹿野搞花千骨，此乃一胜；没有跟罗小黑搞黑猫和他的蓝发师尊，此乃二胜；没有和哪吒搞男同，此乃三胜”",
        "我宣布鹿野是我唯一的姐！太帅了！！！工装裤配T恤，低马尾，非传统女性角色，太帅了5555555希望越来越强，早日拳打无限脚踢各大长老！！！ 以及，真是好多场经费爆炸的打斗啊",
    ]
    negative_reviews = [
        "呃…片方到底懂不懂自己的IP魅力在哪啊！搞什么武器、战争的宏大场面啊，又搞不明白，妥妥露怯！整个剧情就是，稀碎…",
    ]
    return positive_reviews if positive else negative_reviews


@dataclass
class Context:
    user_id: str



@tool
def get_reviews_with_runtime(
    positive: bool, 
    runtime: ToolRuntime[Context] # ToolRuntime 对模型不可见, 最终的实际tool 函数的参数不会包含runtime
) -> list[str]:
    """
    获取罗小黑电影评论列表
    Args:
        positive: 是否获取正面评论, True 表示获取正面评论, False 表示获取负面评论
    Returns:
        评论列表
    """
    print("1.查看对话的状态")
    print(runtime.state["messages"])
    # [HumanMessage(content='请分析罗小黑电影的正面评论原因？'), IMessage(content='我来帮您获取罗小黑电影的正面评论并分析其中的原因。'), ...]

    
    print("2.借助user_id，可以查询user的个人信息")
    user_id = runtime.context.user_id
    print(f"user_id: {user_id}")
    # user_id: user123

    print("3.在长任务中，使用stream_writer反馈进度，通常配合langgraph使用")
    writer = runtime.stream_writer
    writer({"status": "starting", "message": f"正在处理查询"})
    writer({"status": "progress", "message": f"完成50%"})



    positive_reviews = [
        "原来两三岁的小孩也可以不扯女孩裙子啊；原来不整屎尿屁也可以做出让全场大笑的效果啊；原来女角色也可以不穿超短裙高开叉高跟鞋啊；原来男师父女徒弟也可以不暧昧纯师徒情啊；原来一个动画片里正派之间也可以有不同的价值观啊；原来不喊口号不献祭亲朋好友父老乡亲也能表达反战的思想啊。罗小黑你还是太超前了。",
        "瑕不掩瑜。非常好的一点是，一点儿爹味都没有，不judge任何人（妖精），没有任何人（妖精）需要被打败或悔过。这在中国的大型说教重灾区———国漫中已是十分可贵。",
        "“无限虽然爱装逼，但是他没有跟鹿野搞花千骨，此乃一胜；没有跟罗小黑搞黑猫和他的蓝发师尊，此乃二胜；没有和哪吒搞男同，此乃三胜”",
        "我宣布鹿野是我唯一的姐！太帅了！！！工装裤配T恤，低马尾，非传统女性角色，太帅了5555555希望越来越强，早日拳打无限脚踢各大长老！！！ 以及，真是好多场经费爆炸的打斗啊",
    ]
    negative_reviews = [
        "呃…片方到底懂不懂自己的IP魅力在哪啊！搞什么武器、战争的宏大场面啊，又搞不明白，妥妥露怯！整个剧情就是，稀碎…",
    ]
    return positive_reviews if positive else negative_reviews



def test_tool_calling():
    """
    测试工具调用
    """
    # reviews = get_reviews.invoke({"positive": True})
    # print(reviews) # 输出reviews列表: ["原来两三岁...",...]


    tools = [get_reviews]
    tool_by_name = {t.name: t for t in tools}
    model_with_tools = model.bind_tools(tools)
    response = model_with_tools.invoke("请分析罗小黑电影的负面评论原因？") # 返回 AIMessage
    for tool_call in response.tool_calls:
        # 查看函数调用
        print(f"Tool: {tool_call['name']}")
        print(f"Args: {tool_call['args']}")
        # Tool: get_reviews
        # Args: {'positive': False}
        tool = tool_by_name.get(tool_call["name"])
        if tool is None:
            raise ValueError(f"Unknown tool: {tool_call['name']}")
        print(tool.invoke(tool_call["args"]))

def test_tool_calling_2():
    """
    测试工具调用2（测试多轮对话）
    """
    tools = [get_reviews]
    tool_by_name = {t.name: t for t in tools}
    model_with_tools = model.bind_tools(tools)
    prompt = "请分析罗小黑电影的正面评论原因？"
    response = model_with_tools.invoke(prompt)

    tool_messages: list[ToolMessage] = []
    for tool_call in response.tool_calls:
        print(f"Tool: {tool_call['name']}")
        print(f"Args: {tool_call['args']}")
        tool = tool_by_name.get(tool_call["name"])
        if tool is None:
            raise ValueError(f"Unknown tool: {tool_call['name']}")
        reviews = tool.invoke(tool_call["args"])
        tool_messages.append(
            ToolMessage(
                content=json.dumps(reviews, ensure_ascii=False),
                tool_call_id=tool_call["id"],
            )
        )

    final_response = model_with_tools.invoke([HumanMessage(content=prompt), response, *tool_messages])
    print(final_response.content)
    '''输出：
        ## 🎬 正面评论原因分析
        ### 1. **儿童教育价值出色**
        ...
    '''



def test_tool_calling_3():
    """
    测试工具调用3 (仅仅把问题改了，测试并行调用工具)
    """
    tools = [get_reviews]
    tool_by_name = {t.name: t for t in tools}
    model_with_tools = model.bind_tools(tools)
    prompt = "请分析罗小黑电影的正面评论原因和负面评论原因？"
    response = model_with_tools.invoke(prompt)

    tool_messages: list[ToolMessage] = []
    for tool_call in response.tool_calls:
        print(f"Tool: {tool_call['name']}")
        print(f"Args: {tool_call['args']}")
        tool = tool_by_name.get(tool_call["name"])
        if tool is None:
            raise ValueError(f"Unknown tool: {tool_call['name']}")
        reviews = tool.invoke(tool_call["args"])
        tool_messages.append(
            ToolMessage(
                content=json.dumps(reviews, ensure_ascii=False),
                tool_call_id=tool_call["id"],
            )
        )

    final_response = model_with_tools.invoke([HumanMessage(content=prompt), response, *tool_messages])
    print(final_response.content)
    '''输出：
        ## 🎬 正面评论原因分析
        ### 1. **儿童教育价值出色**
        ...
    '''


def test_tool_runtime():
    agent = create_agent(
        model,
        tools=[get_reviews_with_runtime],
        context_schema=Context,
    )
    result = agent.invoke(
        {"messages": [{"role": "user", "content": "请分析罗小黑电影的正面评论原因？"}]},
        context=Context(user_id="user123")
    )

if __name__ == "__main__":
    test_tool_runtime()
