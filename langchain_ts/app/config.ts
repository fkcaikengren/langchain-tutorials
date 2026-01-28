import { z } from "zod";

const envSchema = z.object({
  // Tavily Configuration
  TAVILY_API_KEY: z.string(),

  // SiliconFlow Configuration
  SILICONFLOW_BASE_URL: z.string().url(),
  SILICONFLOW_API_KEY: z.string(),
  DEEPSEEK_API_KEY: z.string(),

  // Model Names
  DS_MODEL: z.string(),
  DSR1_MODEL: z.string(),
  GLM_MODEL: z.string(),
  Qwen3_32B_MODEL: z.string(),

  EMBEDDING_MODEL: z.string(),

  // Milvus Configuration
  MILVUS_ADDRESS: z.string(),
  MILVUS_USERNAME: z.string(),
  MILVUS_PASSWORD: z.string(),
  MILVUS_COLLECTION_NAME: z.string(),
  MILVUS_METRIC_TYPE: z.string(),
  MILVUS_INDEX_TYPE: z.string(),

  // LangSmith
  LANGSMITH_TRACING: z.string().transform((v) => v === "true"),
  LANGCHAIN_TRACING_V2: z.string().transform((v) => v === "true"),
  LANGSMITH_ENDPOINT: z.string().url(),
  LANGSMITH_API_KEY: z.string(),
  LANGSMITH_PROJECT: z.string(),
});

// 解析环境变量
const parsedEnv = envSchema.safeParse(process.env);

if (!parsedEnv.success) {
  console.error("❌ Invalid environment variables:", parsedEnv.error.format());
  process.exit(1);
}

export const settings = {
  tavily_api_key: parsedEnv.data.TAVILY_API_KEY,
  siliconflow_base_url: parsedEnv.data.SILICONFLOW_BASE_URL,
  siliconflow_api_key: parsedEnv.data.SILICONFLOW_API_KEY,
  deepseek_api_key: parsedEnv.data.DEEPSEEK_API_KEY,
  ds_model: parsedEnv.data.DS_MODEL,
  dsr1_model: parsedEnv.data.DSR1_MODEL,
  glm_model: parsedEnv.data.GLM_MODEL,
  qwen3_32b_model: parsedEnv.data.Qwen3_32B_MODEL,
  embedding_model: parsedEnv.data.EMBEDDING_MODEL,
  milvus_address: parsedEnv.data.MILVUS_ADDRESS,
  milvus_username: parsedEnv.data.MILVUS_USERNAME,
  milvus_password: parsedEnv.data.MILVUS_PASSWORD,
  milvus_collection_name: parsedEnv.data.MILVUS_COLLECTION_NAME,
  milvus_metric_type: parsedEnv.data.MILVUS_METRIC_TYPE,
  milvus_index_type: parsedEnv.data.MILVUS_INDEX_TYPE,
  LANGSMITH_TRACING: parsedEnv.data.LANGSMITH_TRACING,
  LANGCHAIN_TRACING_V2: parsedEnv.data.LANGCHAIN_TRACING_V2,
  LANGSMITH_ENDPOINT: parsedEnv.data.LANGSMITH_ENDPOINT,
  LANGSMITH_API_KEY: parsedEnv.data.LANGSMITH_API_KEY,
  LANGSMITH_PROJECT: parsedEnv.data.LANGSMITH_PROJECT,
};



