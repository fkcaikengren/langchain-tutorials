from app.config import settings
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain.messages import HumanMessage, AIMessage, SystemMessage


model = ChatOpenAI(
    model=settings.glm_model,
    base_url=settings.siliconflow_base_url,
    api_key=settings.siliconflow_api_key,
    temperature=0.9,
    max_tokens=3000,
    timeout=60,
)

def test_invoke():
    """
    invoke 调用
    """
    # res = model.invoke("Translate 'I love programming' into Chinese.")
    # 等价于
    # res = model.invoke([{"role": "user", "content": "Translate 'I love programming' into Chinese."}])
    # 等价于
    # res = model.invoke([HumanMessage(content="Translate 'I love programming' into Chinese.")])
    # print(res)
    """
    返回：AIMessage
    {
        "content": "我喜欢编程。",
        "response_metadata": {...},
        "id": "...",
        ...
    }
    """
    # 通过管道符 | 链接模型和解析器，返回一个增强的链。 新的链调用模型的结果 会从 AIMessage 转换为字符串
    # 文档参考：https://reference.langchain.com/python/langchain_core/output_parsers/#langchain_core.output_parsers.JsonOutputParser.assign
    chain = model | StrOutputParser() # 这就是 LCEL (LangChain Expression Language)
    ans = chain.invoke("Translate 'I love programming' into Chinese.")
    print(ans)
    """
    输出：我喜欢编程。
    """


def test_stream():
    """
    Stream 调用 (流式输出)
    """
    chain = model | StrOutputParser()
    messages = [
        ["system", "You are a helpful translator. Translate the user sentence to Chinese."],
        ["human", "I love programming."]
    ]
    stream = chain.stream(messages) # 返回一个迭代器
    for chunk in stream: 
        print(chunk, end="\n", flush=True)
    """
    输出：
        我很
        喜欢
        编程。
    """


def test_batch():
    """
    # Batch 调用 (并行执行多个请求)
    数组的每一项是一个请求, 请求是并行的
    """
    chain = model | StrOutputParser()
    batches = chain.batch([
        [
            {"role": "system", "content": "You are a helpful translator. Translate the sentence to Chinese."},
            {"role": "human", "content": "I love programming."}
        ],
        [
            {"role": "human", "content": "100字内，介绍下langchain。"}
        ],
    ])
    for i, res in enumerate(batches):
        print(f"Result {i+1}:\n{res}\n")

    """
    输出：
        Result 1:
        我非常热爱编程。

        Result 2:
        LangChain是一个开源框架，旨在简化基于大型语言模型（LLM）的应用程序开发。它通过提供模块化组件和工具链，帮助开发者轻松连接LLM与外部数据源、API或计算资源，实现数据感知和代理式交互应用。核心功能包括Prompt模板化、记忆管理、链式调用及多工具集成，大幅提升开发效率。
    """

    # 默认情况下， batch() 仅会返回整个批次的最终输出。如果你希望随着每个单独输入生成完成时接收其输出，可以使用 batch_as_completed() 进行结果流式传输：
    # batches = chain.batch_as_completed([
    #     [
    #         {"role": "system", "content": "You are a helpful translator. Translate the sentence to Chinese."},
    #         {"role": "human", "content": "I love programming."}
    #     ],
    #     [
    #         {"role": "human", "content": "100字内，介绍下langchain。"}
    #     ],
    # ])

if __name__ == "__main__":
    test_batch()
