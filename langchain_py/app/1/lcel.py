from app.config import settings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate,AIMessagePromptTemplate, ChatPromptTemplate
from langchain_core.runnables import RunnableSequence, RunnablePassthrough, RunnableLambda, RunnableBranch
from langchain_core.output_parsers import StrOutputParser
from langchain.messages import HumanMessage, AIMessage, SystemMessage
from pydantic import BaseModel
from typing import Literal


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



def test_runnable_sequence():

    prompt = ChatPromptTemplate(
        [
            ("system", "你是一个{type}，请根据用户的提问进行回答"),
            ("system", "{instruction}"),
            ("human", "{question}"),
        ]
    )

    def classifier(input: dict) -> dict:
        question: str = input["question"]
        instruction: str = input["instruction"]
        type_value = "科普专家" if "科普" in question else "智能助手"
        return {
            "type": type_value,
            "question": question,
            "instruction": instruction,
        }
    # RunnableLambda将classifier包装成一个Runnable，使它可以在LCEL中使用
    # chain = RunnableLambda(classifier) | prompt | model | StrOutputParser()
    # 等价于
    chain = RunnableSequence(RunnableLambda(classifier), prompt, model, StrOutputParser())  
    ans = chain.invoke(
        {
            "question": "科普，鲸鱼是哺乳动物么？只需要回答是或不是",
            "instruction": "用中文回答",
        }
    )
    print(ans)
    # 是


def test_runnable_branch():
    class ClassifyResult(BaseModel):
        type: Literal["科普", "编程", "其他"]

    structured_model = model.with_structured_output(ClassifyResult)

    science_prompt = ChatPromptTemplate(
        [
            ("system", "你是科普专家，通俗准确、简洁回答。"),
            ("human", "{question}"),
        ]
    )
    science_expert = RunnableSequence(science_prompt, model, StrOutputParser())

    code_prompt = ChatPromptTemplate(
        [
            ("system", "你是编程专家，提供代码或技术解答。"),
            ("human", "{question}"),
        ]
    )
    code_expert = RunnableSequence(code_prompt, model, StrOutputParser())

    general_prompt = ChatPromptTemplate(
        [
            ("system", "你是智能助手，简洁中文回答。"),
            ("human", "{question}"),
        ]
    )
    general_expert = RunnableSequence(general_prompt, model, StrOutputParser())


    classifier = RunnablePassthrough.assign(
        # classify_result 会 赋值给 input["classify_result"]
        classify_result=lambda input: structured_model.invoke(
            f"请判断以下问题的类型，并只返回JSON：{{\"type\": \"科普\"|\"编程\"|\"其他\"}}。问题：{input['question']}"
        )
    )
    debug_runnable = RunnableLambda(
        lambda input: (print("中间结果:", input), input)[1]
    )

    # RunnableBranch 将从三个分支中选取一个，根据lambda函数的返回值判断选择哪一个。
    branch = classifier | debug_runnable | RunnableBranch(
        (
            lambda input: getattr(input.get("classify_result"), "type", None) == "科普",
            science_expert,
        ),
        (
            lambda input: getattr(input.get("classify_result"), "type", None) == "编程",
            code_expert,
        ),
        general_expert,
    )

    result = branch.invoke({"question": "简单回答下，鲸鱼是哺乳动物吗？"})
    print(result)
    # 中间结果: {'question': '简单回答下，鲸鱼是哺乳动物吗？', 'classify_result': ClassifyResult(type='科普')}
    # 是的，**鲸鱼是哺乳动物**，而不是鱼类。
    


if __name__ == "__main__":
    # test_prompt_template()
    # test_output_parser()
    # test_lcel()
    # test_runnable_sequence()
    test_runnable_branch()
