
from langchain.agents import create_agent
from app.config import settings
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field, RootModel
from langchain_core.globals import set_debug
from langchain_core.output_parsers import PydanticOutputParser,CommaSeparatedListOutputParser

# 开启全局详细模式
set_debug(True)


model = ChatOpenAI(
    model=settings.glm_model,
    base_url=settings.siliconflow_base_url,
    api_key=settings.siliconflow_api_key,
    temperature=0.9,
    max_tokens=5000,
    timeout=60,
)

class Movie(BaseModel):
    """电影的相关信息"""
    title: str = Field(..., description="电影名称")
    year: int = Field(..., description="电影上映时间")
    director: str = Field(..., description="电影的导演")
    rating: float = Field(..., description="电影的豆瓣评分")


class DevProcessList(RootModel[list[str]]):
    """按顺序的软件开发流程字符串列表"""

def test_structure_class():
    model_with_structure = model.with_structured_output(Movie)
    response = model_with_structure.invoke(
        [{"role": "user", "content": "介绍下电影《罗小黑战记2》，获取title、year、director、rating信息"}], 
    )
    print(type(response))
    print(response)


def test_structure_list():
    # 使用LLM provider API 强制结构化输出
    model_with_structure = model.with_structured_output(DevProcessList)
    # 使用langchain自己的解析器 获取结构化输出（可靠性一般）
    # model_with_structure = model | PydanticOutputParser(pydantic_object=DevProcessList)
    response = model_with_structure.invoke(
        [{"role": "user", "content": "软件开发的流程是？请给我一个有顺序的字符串列表"}], 
    )
    print(type(response)) # <class '__main__.DevProcessList'>
    print(response.model_dump()) # ['需求分析', '系统设计', '编码实现', '软件测试', '部署发布', '运维维护']

    # P.S. 使用CommaSeparatedListOutputParser （复杂格式准确率相对不高，但更快更节省token）
    # model_with_structure = model | CommaSeparatedListOutputParser()
    # response = model_with_structure.invoke(
    #     [{"role": "user", "content": "软件开发的流程是？请给我一个有顺序的字符串列表"}], 
    # )
    # print(response) ['以下是标准软件开发流程（SDLC）的字符串列表：', '1. 需求分析', '2. 系统设计', '3. 开发实施', '4. 软件测试', '5. 部署上线', '6. 运维与迭代']
    

if __name__ == "__main__":
    # test_structure_class()
    test_structure_list()
