import { settings } from "@/config";
import { MemorySaver } from "@langchain/langgraph";
import { ChatOpenAI } from "@langchain/openai";
import type { RunnableConfig } from "@langchain/core/runnables";
import { createAgent, tool, toolStrategy } from "langchain";
import { z } from "zod";

/*
标准的agent创建和使用流程：
1.定义提示词
2.定义工具
3.构建chat model
4.结构化输出
5.memory
6.创建agent
*/

// 1.定义提示词
const SYSTEM_PROMPT =
  "你是天气助手,如果被问到天气问题，请先确定地点，然后调用相关工具获取实际天气";

// 2.定义工具
const get_weather_for_location = tool(({ city }: { city: string }) => {
  return `${city}总是晴日`;
}, {
  name: "get_weather_for_location",
  description: "获取指定城市的天气",
  schema: z.object({
    city: z.string(),
  }),
});

const get_user_location = tool(
  async (_: Record<string, never>, config?: RunnableConfig) => {
    const user_id = String(
      (config?.configurable as Record<string, unknown> | undefined)?.user_id ??
        "",
    );
    return user_id === "1" ? "北京" : "上海";
  },
  {
    name: "get_user_location",
    description: "根据用户ID获取用户位置",
    schema: z.object({}),
  },
);

// 3.构建chat model
// 符合openai规范的api,可以使用 @langchain/openai。我们使用 SiliconFlow 提供的 GLM-4.7 模型
// 注意：尽量使用一些新模型，一些旧模型可能会存在一些特性不支持
const model = new ChatOpenAI({
  model: settings.glm_model,
  apiKey: settings.siliconflow_api_key,
  configuration: {
    baseURL: settings.siliconflow_base_url,
  },
  temperature: 0.9,
  maxTokens: 5000,
  timeout: 60000,
});

// 4.结构化输出
// dataclass 和 Pydantic 都是支持的，用来定义结构化输出的格式。
const ResponseFormat = z.object({
  // 一语双关的回答 (必要)
  punny_response: z.string().describe("一语双关的回答（必要）"),
  // 和天气相关的信息点（可选）
  weather_conditions: z
    .string()
    .nullable()
    .optional()
    .describe("和天气相关的信息点（可选）"),
  // 字符串，用于描述响应的长度，取值为"short"、"medium"、"long"之一
  length: z
    .enum(["short", "medium", "long"])
    .default("short")
    .describe('响应长度，取值为"short"、"medium"、"long"之一'),
}).describe("agent的响应格式");

// 5.memory
const checkpointer = new MemorySaver();

// 6.创建agent
const agent = createAgent({
  model,
  systemPrompt: SYSTEM_PROMPT,
  tools: [get_user_location, get_weather_for_location],
  responseFormat: toolStrategy(ResponseFormat),
  checkpointer,
  // debug: true, # 开启debug模式，会打印出agent的运行过程
});

async function main() {
  // `thread_id`一次会话的唯一标识符
  const config = { configurable: { thread_id: "1", user_id: "1" } };

  const response1 = await agent.invoke(
    { messages: [{ role: "user", content: "今天天气如何？" }] },
    config,
  );
  console.log(response1.structuredResponse);
//   {
//     punny_response: "北京总是晴日，今天也要保持阳光好心情！",
//     weather_conditions: "晴朗",
//     length: "short",
//   }

  // 注意：我们可以用同一个`thread_id`继续这个对话.
  const response2 = await agent.invoke(
    { messages: [{ role: "user", content: "thank you!" }] },
    config,
  );
  console.log(response2.structuredResponse);
// {
//   punny_response: "不客气！祝你有个阳光灿烂的一天！",
//   length: "short",
// }
}

if (import.meta.main) {
  await main();
}
