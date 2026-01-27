
from app.config import settings
from langchain_openai import ChatOpenAI
from langchain.chat_models import init_chat_model
from langchain_deepseek import ChatDeepSeek

def teset_deepseek_model():
    """
    测试deepseek模型，需要先在config中配置好模型的参数
    """
    # 使用 init_chat_model 或者 ChatDeepSeek 
    
    # model = init_chat_model(
    #     "deepseek-chat", # 指 DeepSeek-V3.2 (名称参考：https://api-docs.deepseek.com/zh-cn/quick_start/pricing)
    #     # model_provider="deepseek", #可省略
    #     api_key=settings.deepseek_api_key,
    #     temperature=0.9,
    #     max_tokens=1000,
    #     timeout=60,
    # )
    model = ChatDeepSeek(
        model="deepseek-chat",
        api_key=settings.deepseek_api_key,
        temperature=0.9,
        max_tokens=1000,
        timeout=60,
    )
    response = model.invoke("你好")
    print(response.content)

def test_third_part_model():
    """
    测试第三方模型，只要符合openai规范的api，都可以用ChatOpenAI来创建该模型的实例
    """
    model = ChatOpenAI(
        model=settings.ds_model,
        base_url=settings.siliconflow_base_url,
        api_key=settings.siliconflow_api_key,
        temperature=0.9,
        max_tokens=1000,
        timeout=60,
    )

    response = model.invoke("你好")
    print(response.content)


if __name__ == "__main__":
    teset_deepseek_model()
    # test_third_part_model()
