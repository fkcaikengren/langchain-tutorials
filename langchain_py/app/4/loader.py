import app.config
from pathlib import Path
from typing import List
from langchain_core.documents import Document
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    JSONLoader,
    DirectoryLoader,
    CSVLoader,
    WebBaseLoader,
    GithubFileLoader,
)

"""
常用 Document Loader 示例
参考：https://python.langchain.com/docs/integrations/document_loaders
"""


# ================================== 公共 ==================================
base_dir = Path(__file__).parent

# ================================== 公共 ==================================

def test_pypdf_loader() -> List[Document]:
    """使用 PyPDFLoader 加载本地 PDF 示例。"""

    pdf_path = base_dir / "files" / "keyboard_instructions.pdf"
    # 需要安装 pypdf 库
    loader = PyPDFLoader(str(pdf_path))
    docs = loader.load()

    
    print(f"加载到的文档页数: {len(docs)}")
    if docs:
        first_doc = docs[0]
        print("第一页内容前 300 字:\n")
        print(first_doc.page_content[:300])

        # print(f"元数据 meta: {first_doc.metadata}")

    return docs
    # 一页就是一个文档。按页拆分，并携带页码等元数据
    # PDF loader 是抽取文本，并不保留原始格式，对于一些表格等会导致信息丢失或不对齐
    """输出
        加载到的文档页数: 4
        第一页内容前 300 字:

        自定义灯光颜色
        ①： Fn+Tab,进入录制模式， 1-5灯亮 （提示可以录制5组灯光,同时提示保存在第几组） 。
        ②： 按下数字选择组别， 1： 第一组、 2： 第二组、 3： 第三组、 4： 第四组、 5： 第五组。
        ③： 按下所需按键， 点按切换灯光状态。 切换顺序为亮、 灭。
            保存时Fn键灯光的状态会切换一次， 请把Fn键灯光的状态调到你想要的前一个状态。
        ④： Fn+Tab， 进行保存。
        ⑤： Fn+→， 展示自定义灯效。
            未录制灯光的情况下， Fn+→， 第一组灯光默认为1、 Q、 A、 Z；
                
    """



def test_text_loader() -> List[Document]:
    """使用 TextLoader 加载本地 TXT 示例。"""

    txt_path = base_dir / "files" / "keyboard_guide.txt"

    loader = TextLoader(str(txt_path))
    docs = loader.load()

    print(f"加载到的文档数: {len(docs)}")
    if docs:
        first_doc = docs[0]
        print("第一行内容前 50 字:\n")
        print(first_doc.page_content[:50])

    return docs
    # 整个文件就是一个文档
    """输出
        加载到的文档数: 1
        第一行内容前 50 字:

        Esc组合键功能：
        Shift+Esc 对应 ～
        Fn+Esc 对应 `

        Fn组合键功能：
        Fn+
    """


def test_csv_loader() -> List[Document]:
    """使用 CSVLoader 加载本地 CSV 示例。"""

    csv_path = base_dir / "files" / "keyboard_guide.csv"

    loader = CSVLoader(str(csv_path), encoding="utf-8")
    docs = loader.load()

    print(f"加载到的行数: {len(docs)}\n")
    if docs:
        first_doc = docs[0]
        print("第一行内容:\n")
        print(first_doc.page_content)
        print('\n')
        first_doc = docs[1]
        print("第二行内容:\n")
        print(first_doc.page_content)

    return docs
    # 每一行就是一个文档
    """输出
        加载到的行数: 10

        第一行内容:

        功能分类: Esc功能
        按键组合: Shift+Esc
        功能: ~


        第二行内容:

        功能分类: 
        按键组合: Fn+Esc
        功能: `
    """


def test_json_loader() -> List[Document]:
    """使用 JSONLoader 加载本地 JSON 示例。"""

    json_path = base_dir / "files" / "keyboard_guide.json"
    # 需要安装 jq 库
    loader = JSONLoader(
        file_path=str(json_path),
        jq_schema=".[].list[].[\"功能\"]",
        text_content=False,
    )
    """jq_schema 参数含义
        .：表示 JSON 数据的根节点。
        []：表示遍历一个数组。如果你的 JSON 最外层是一个数组，这一步会展开这个数组。
        .list：在展开后的每一个对象中，寻找键名为 "list" 的字段。
        []：表示 "list" 字段的值也是一个数组，再次对其进行遍历。
        .功能：在 list 数组的每一个元素中，提取键名为 "功能" 的值。
    """
    docs = loader.load()

    print(f"加载到的按键功能条目数: {len(docs)}")
    for doc in docs:
        print(doc.page_content)

    return docs
    """输出
        加载到的按键功能条目数: 10
        ~
        `
        F行(F1至F12)
        数字行(1至=)
        Print Screen
        Scroll Lock
        Pause
        Home
        End
        Application Key
    """


def test_webbase_loader() -> List[Document]:
    """使用 WebBaseLoader 加载网页示例。"""

    loader = WebBaseLoader("https://juejin.cn/post/7599708108622364723")
    docs = loader.load()

    print(f"加载到的文档数: {len(docs)}")
    if docs:
        first_doc = docs[0]
        print("网页内容前 100 字:\n")
        print(first_doc.page_content[:100])

    return docs
    """
    加载到的文档数: 1
    网页内容前 100 字:


    【Python版 2026 从零学Langchain  1.x】（一）快速开始和LCEL本文介绍了LLM的背景知识（Tr - 掘金
    """


def test_github_file_loader() -> List[Document]:
    """使用 GithubFileLoader 加载 GitHub 仓库文件示例。"""

    # 需要安装 beautifulsoup4 库
    # 不要忘记配置 GITHUB_PERSONAL_ACCESS_TOKEN 环境变量
    loader = GithubFileLoader(
        repo="fkcaikengren/langchain-tutorials",
        branch="main",
        file_filter=lambda file_path: file_path.endswith(".md"), # 仅加载 Markdown 文件
    )
    docs = loader.load()

    print(f"加载到的文档数: {len(docs)}")
    if docs:
        first_doc = docs[0]
        print("第一个文档的元数据:\n")
        print(first_doc.metadata)
        print("第一个文档内容前 100 字:\n")
        print(first_doc.page_content[:100])

    return docs
    """
    加载到的文档数: 6
    """

def test_directory_loader() -> List[Document]:
    """使用 DirectoryLoader 加载本地目录示例。"""

    dir_path = base_dir / "files"

    # 在底层，默认情况下使用 UnstructuredFileLoader，可以通过loader_cls替换. 
    # UnstructuredFileLoader需要安装unstructured。 很遗憾安装时报错，RuntimeError: Cannot install on Python version 3.12.12; only versions >=3.6,<3.10 are supported
    # 通常是按文件类型筛选，如 **/*.pdf 或 **/*.txt，要加载多种文件类并指定loader_cls，则需要构建多个DirectoryLoader
    loader = DirectoryLoader(
        str(dir_path),
        glob="**/*.pdf", #筛选
        loader_cls=PyPDFLoader,
        show_progress=True,
        use_multithreading=True,
    )
    docs = loader.load()


    print(f"加载到的文档数: {len(docs)}")
    # 加载到的文档数: 5  


if __name__ == "__main__":
    # test_pypdf_loader()
    # test_text_loader()
    # test_csv_loader()
    # test_json_loader()

    # test_webbase_loader()
    # test_github_file_loader()

    test_directory_loader()


    
