import numpy as np
from app.config import settings
from langchain_openai import OpenAIEmbeddings

"""
OpenAIEmbeddings 模型测试
参考：https://docs.langchain.com/oss/python/integrations/text_embedding/openai#embed-single-texts
"""

# ================================== 公共 ==================================
def cosine_similarity(vec1:list[float], vec2:list[float]) -> float:
    """计算两个向量的余弦相似度"""
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)   
    dot = np.dot(vec1, vec2)
    return dot / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

embeddings = OpenAIEmbeddings(
        model=settings.embedding_model,
        base_url=settings.siliconflow_base_url,
        api_key=settings.siliconflow_api_key,
    )

# ================================== 公共 ==================================

def test_cosine_similarity():
    
    similarity = cosine_similarity([1, 2, 3], [4, 5, 6])
    print("向量相似度:", similarity)
    # 向量相似度: 0.9746318461970762


def test_embedding_model():
    # 直接使用OpenAI提供的Embedding模型
    # embeddings = OpenAIEmbeddings(
    #     model="text-embedding-3-large",
    # )
    # 这里使用第三方Embedding模型（需要配置baseURL）
    embeddings = OpenAIEmbeddings(
        model=settings.embedding_model,
        base_url=settings.siliconflow_base_url,
        api_key=settings.siliconflow_api_key,
        # dimensions=1024 # 1024, 1536, 2560 (Qwen/Qwen3-Embedding-4B 最多支持到2560维)
    )

    text = "你好"
    single_vector = embeddings.embed_query(text)
    print(f"向量长度: {len(single_vector)}") # 2560
    print(single_vector[:20])  
    # 向量长度: 2560
    # [-0.0006555612199008465, -0.01359222736209631, 0.0036308004055172205, -0.028798144310712814, -0.0039100926369428635, 0.038480278104543686, 0.005647911690175533, 0.12015777081251144, 0.04369373619556427, 0.10873781889677048, -0.01986078917980194, -0.021474478766322136, 0.015454176813364029, -0.05014849081635475, 0.1668306291103363, 0.025694895535707474, 0.012350928038358688, 0.031777262687683105, 0.12214384973049164, 0.0035687354393303394]

def test_embedding_similarity():
    """
    Embedding模型计算的向量，相似度计算的结果测试，评估embedding模型的相似度计算能力
    """

    # 完全相似
    similarity = cosine_similarity(
        embeddings.embed_query("手机"), 
        embeddings.embed_query("手机")
    )
    print("手机和手机的相似度:", similarity)

    # 苹果手机和乔布斯 有很高的相似度
    similarity = cosine_similarity(
        embeddings.embed_query("苹果手机"), 
        embeddings.embed_query("乔布斯")
    )
    print("苹果手机和乔布斯的相似度:", similarity)

    # 苹果手机和汽车 相似度较低
    similarity = cosine_similarity(
        embeddings.embed_query("苹果手机"), 
        embeddings.embed_query("汽车")
    )
    print("苹果手机和汽车的相似度:", similarity)

    # 测试两段话的相似度
    similarity = cosine_similarity(
        embeddings.embed_query("乔布斯是苹果公司的创始人，苹果手机是苹果公司生产的智能手机"), 
        embeddings.embed_query("乔布斯发明了苹果手机")
    )
    print("两段话的相似度:", similarity)

    # 手机和手机的相似度: 0.9999999999999999
    # 苹果手机和乔布斯的相似度: 0.5394882107942822
    # 苹果手机和汽车的相似度: 0.45631589739439193
    # 两段话的相似度: 0.7551136748883456


def test_create_embeddings():
    """
    embed_documents 同时创建多个向量
    """
    text1 = "你好"
    text2 = (
        "Langchain是一个用于构建基于LLM的应用程序的框架"
    )
    two_vectors = embeddings.embed_documents([text1, text2])
    print(len(two_vectors))
    # 2


if __name__ == "__main__":
    # test_cosine_similarity()
    # test_embedding_model()
    # test_embedding_similarity()
    test_create_embeddings()
