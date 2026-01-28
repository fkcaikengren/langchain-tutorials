from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

load_dotenv()

class AppSettings(BaseSettings):

    # Tavily Configuration
    tavily_api_key: str = Field(..., alias='TAVILY_API_KEY')

    # SiliconFlow Configuration
    siliconflow_base_url: str = Field(..., alias='SILICONFLOW_BASE_URL')
    siliconflow_api_key: str = Field(..., alias='SILICONFLOW_API_KEY')
    deepseek_api_key: str = Field(..., alias='DEEPSEEK_API_KEY')

    # Model Names
    ds_model: str = Field(..., alias='DS_MODEL')
    dsr1_model: str = Field(..., alias='DSR1_MODEL')
    glm_model: str = Field(..., alias='GLM_MODEL')
    qwen3_32b_model: str = Field(..., alias='Qwen3_32B_MODEL')

    embedding_model: str = Field(..., alias='EMBEDDING_MODEL')

    # Milvus Configuration
    milvus_address: str = Field(..., alias='MILVUS_ADDRESS')
    milvus_username: str = Field(..., alias='MILVUS_USERNAME')
    milvus_password: str = Field(..., alias='MILVUS_PASSWORD')
    milvus_collection_name: str = Field(..., alias='MILVUS_COLLECTION_NAME')
    milvus_metric_type: str = Field(..., alias='MILVUS_METRIC_TYPE')
    milvus_index_type: str = Field(..., alias='MILVUS_INDEX_TYPE')



    # langsmith 
    LANGSMITH_TRACING: bool = Field(..., alias='LANGSMITH_TRACING')
    LANGCHAIN_TRACING_V2: bool = Field(..., alias='LANGCHAIN_TRACING_V2')
    LANGSMITH_ENDPOINT: str = Field(..., alias='LANGSMITH_ENDPOINT')
    LANGSMITH_API_KEY: str = Field(..., alias='LANGSMITH_API_KEY')
    LANGSMITH_PROJECT: str = Field(..., alias='LANGSMITH_PROJECT')



    model_config = SettingsConfigDict(
        env_file=".env",               # 指定 .env 文件路径
        env_file_encoding="utf-8",
        case_sensitive=False,          # 环境变量不区分大小写（推荐）
        extra="ignore"                 # 忽略未定义的变量
    )

settings = AppSettings()
