from app.config import settings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate,AIMessagePromptTemplate, ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel, RunnableLambda
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

def test_prompt_template():
    """
    测试提示模板:
        SystemMessagePromptTemplate
        HumanMessagePromptTemplate
        AIMessagePromptTemplate
        ChatPromptTemplate
    """
    h_prompt = HumanMessagePromptTemplate.from_template(
        "你是一个专业的翻译。请将以下文本从英文翻译为中文：\n {input}"
    )
    h_message = h_prompt.format(input="I love programming.")
    print('h_message 是 HumanMessage 类型吗？', isinstance(h_message, HumanMessage)) # True
    print(h_message)
    # 输出
    # HumanMessage(content='你是一个专业的翻译。请将以下文本从英文翻译为中文：\n I love programming.', additional_kwargs={}, response_metadata={})
    

    c_prompt = ChatPromptTemplate(
        [
            ("system", "你是一个{role}。"),
            ("human", "请将以下文本从英文翻译为中文：\n {user_input}"),
        ]
    )
    
    # ChatPromptTemplate 是runnable, 调用invoke 。 返回 Message列表
    c_messages = c_prompt.invoke({
        "role": "专业的翻译", "user_input": "I love programming."
    })
    print(c_messages)
    # 输出：
    # ChatPromptValue(
    #    messages=[
    #        SystemMessage(content='你是一个专业的翻译。', additional_kwargs={}, response_metadata={}),
    #        HumanMessage(content='请将以下文本从英文翻译为中文：\n I love programming.', additional_kwargs={}, response_metadata={})
    #    ]
    # )

    # ！注意：format 方法返回的是一个字符串列表
    # c_messages = c_prompt.format(role="专业的翻译", user_input="I love programming.")
    # print(type(c_messages[0])) # <class 'str'>
    # print(type(c_messages[1])) # <class 'str'>
    # print(c_messages) 


def test_output_parser():
    """
    测试输出解析器:
        StrOutputParser
    """
    parser = StrOutputParser()
    ans = parser.invoke("翻译 'I love programming' 成中文。")
    print(ans)
    # 输出：
    # 翻译 'I love programming' 成中文。


def test_lcel():
    """
    测试 LCEL (LangChain Expression Language)
    """
    c_prompt = ChatPromptTemplate(
        [
            ("system", "你是一个{role}。"),
            ("human", "请将以下文本从英文翻译为中文：\n {user_input}"),
        ]
    )

    chain = c_prompt | model | StrOutputParser()
    ans = chain.invoke({
        "role": "专业的翻译", "user_input": "I love programming."
    })
    print(ans)
    # 输出：
    # 我热爱编程。

def test_runnable():
    pass;
if __name__ == "__main__":
    # test_prompt_template()
    # test_output_parser()
    # test_lcel()
    test_runnable()
