import { settings } from "@/config";
import { AIMessage, HumanMessage, SystemMessage } from "@langchain/core/messages";
import type { BaseMessage } from "@langchain/core/messages";
import { ChatOpenAI } from "@langchain/openai";
import { SqliteSaver } from "@beshkenadze/langgraph-checkpoint-libsql";
import { MemorySaver } from "@langchain/langgraph";
import { createAgent, summarizationMiddleware } from "langchain";

const systemPrompt = "你是一个人工智能助手";

function buildModels() {
  const baseConfig = {
    apiKey: settings.siliconflow_api_key,
    configuration: {
      baseURL: settings.siliconflow_base_url,
    },
    timeout: 60_000,
  } as const;

  const agentModel = new ChatOpenAI({
    ...baseConfig,
    model: settings.qwen3_32b_model,
    temperature: 0.7,
    maxTokens: 2000,
  });

  const summarizerModel = new ChatOpenAI({
    ...baseConfig,
    model: settings.qwen3_32b_model,
    temperature: 0.2,
    maxTokens: 512,
  });

  return { agentModel, summarizerModel };
}

async function testSummarizationShortMemory() {
  const { agentModel, summarizerModel } = buildModels();
  const checkpointer = new MemorySaver();
  const agent = createAgent({
    model: agentModel,
    tools: [],
    systemPrompt,
    middleware: [
      summarizationMiddleware({
        model: summarizerModel,
        trigger: { messages: 4 }, // 也可以按 tokens 来
        keep: { messages: 4 }, // 也可以按 tokens 来
        summaryPrefix: "对话摘要：",
        summaryPrompt:
          "请将以下对话历史压缩成简短的中文摘要，保留关键信息（事实、偏好、约束、决定、结论）：\n{messages}",
      }),
    ],
    checkpointer,
  });

  const userInputs = [
    "我叫小明，住在北京。",
    "请记住我更喜欢用中文回答。",
    "我这周想做一个 LangChain 的学习计划。简短控制在100字",
    "计划要按天拆分，每天不超过1小时。简短控制在100字",
    "顺便提醒我：周三晚上要健身。最后请把所有安排再用要点总结一次。",
  ];

  const config = { configurable: { thread_id: "short-memory-demo" } };

  for (let idx = 0; idx < userInputs.length; idx += 1) {
    const text = userInputs[idx] || '';
    const r = await agent.invoke({ messages: [{ role: "user", content: text }] }, config);
    const messages = r.messages;
    const last = messages.at(-1);

    console.log(
      `\n[Turn ${idx + 1}] 当前上下文消息数：${messages.length}（trigger=4, keep=4）\n`,
    );
    console.log(`用户：${text}`);
    console.log(`助手：${String(last?.content ?? "")}`);
  }
}

async function testSqliteSaver() {
  const { agentModel } = buildModels();

  const checkpointer = SqliteSaver.fromConnString("file:checkpoints.db");
  const agent = createAgent({
    model: agentModel,
    tools: [],
    systemPrompt,
    checkpointer,
  });

  const config = { configurable: { thread_id: "test_sqlite_saver" } };

  const r1 = await agent.invoke(
    { messages: [{ role: "user", content: "你好，我叫“疯狂踩坑人”" }] },
    config,
  );
  console.log("[assistant]", String(r1.messages.at(-1)?.content ?? ""));
  // [assistant] 
  // 我是你的AI助手！不过“疯狂踩坑人”这个名字挺有...

  const r2 = await agent.invoke(
    { messages: [{ role: "user", content: "请问我叫什么名字？" }] },
    config,
  );
  console.log("[assistant]", String(r2.messages.at(-1)?.content ?? ""));
  // [assistant] 
  // 你叫“疯狂踩坑人”呀...
}

if (import.meta.main) {
  // await testSummarizationShortMemory();
  await testSqliteSaver();
}

