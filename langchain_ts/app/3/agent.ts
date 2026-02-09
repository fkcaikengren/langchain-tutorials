import { settings } from "@/config";
import { HumanMessage, ToolMessage } from "@langchain/core/messages";
import type { BaseMessage } from "@langchain/core/messages";
import { MemorySaver } from "@langchain/langgraph";
import { ChatOpenAI } from "@langchain/openai";
import { createAgent, createMiddleware, tool, toolStrategy } from "langchain";
import type { ToolRuntime } from "langchain";
import { z } from "zod";

const contextSchema = z.object({
  userId: z.string().describe("用户ID"),
});

const compareTwoNumbers = tool(
  ({ a, b }: { a: number; b: number }) => {
    console.log("TOOL CALLED: compare_two_numbers", { a, b });
    if (a > b) return 1;
    if (a < b) return -1;
    return 0;
  },
  {
    name: "compare_two_numbers",
    description: "比较两个数字a，b的大小。比较结果：如果a>b返回1，a<b返回-1，a=b返回0",
    schema: z.object({
      a: z.number().describe("第一个数字 a"),
      b: z.number().describe("第二个数字 b"),
    }),
  },
);

const getUserLocation = tool(
  async (_: Record<string, never>, runtime: ToolRuntime<any, typeof contextSchema>) => {
    const userId = runtime.context.userId;
    return userId === "1" ? "北京" : "上海";
  },
  {
    name: "get_user_location",
    description: "根据用户ID获取用户位置",
    schema: z.object({}),
  },
);

/**
 * 定义一个 ChatOpenAI 模型
 * @param model 模型名称
 * @param maxTokens 最大 token 数
 * @returns ChatOpenAI 模型实例
 */
function createChatModel(model: string, maxTokens: number) {
  return new ChatOpenAI({
    model,
    apiKey: settings.siliconflow_api_key,
    configuration: {
      baseURL: settings.siliconflow_base_url,
    },
    temperature: 0.9,
    maxTokens,
    timeout: 60_000,
  });
}
// 定义三个模型
const glmModel = createChatModel(settings.glm_model, 10_000);
const dsModel = createChatModel(settings.ds_model, 10_000);
const qwenRouterModel = createChatModel(settings.qwen3_32b_model, 64);


/**
 * 从输入中提取最新的用户文本
 * @param input 输入内容，可能是字符串或包含消息数组的对象
 * @returns 最新的用户文本内容
 */
function extractLatestUserText(input: unknown): string {
  if (typeof input === "string") return input;

  const messages: BaseMessage[] | undefined = Array.isArray(input)
    ? (input as BaseMessage[])
    : (input as { messages?: BaseMessage[] } | null | undefined)?.messages;

  if (!messages?.length) return "";
  for (let i = messages.length - 1; i >= 0; i -= 1) {
    const msg = messages[i];
    if (msg instanceof HumanMessage) return String(msg.content);
  }
  return String(messages.at(-1)?.content ?? "");
}


/**
 * 判断用户问题的复杂度
 * @param userText 用户输入的文本
 * @returns 问题复杂度，"simple" 或 "complex"
 */
async function judgeComplexity(userText: string): Promise<"simple" | "complex"> {
  const response = await qwenRouterModel.invoke([
    {
      role: "system" as const,
      content:
        "你是问题复杂度分类器。根据用户问题判断复杂度：\n- simple：单一事实/常识问答、简单翻译/润色、很短的直接回答、无需多步推理或设计。\n- complex：需要多步推理、方案设计/架构、长文写作、复杂代码/调试、严谨数学推导、对比权衡。\n只输出：simple 或 complex。",
    },
    { role: "user" as const, content: userText },
  ]);

  const text = String(response.content ?? "").trim().toLowerCase();
  console.log("判断复杂度：", text);
  if (text === "simple" || text.includes("simple") || text.includes("简单")) return "simple";
  if (text === "complex" || text.includes("complex") || text.includes("复杂")) return "complex";
  return "complex";
}

const dynamicModelMiddleware = createMiddleware({
  name: "DynamicModelMiddleware",
  wrapModelCall: async (request, handler) => {
    const userText = extractLatestUserText({ messages: (request as any)?.messages });
    const complexity = await judgeComplexity(userText);
    const model = complexity === "simple" ? dsModel : glmModel;
    request.model = model
    return handler(request);
  },
});

const handleToolErrorsMiddleware = createMiddleware({
  name: "HandleToolErrorsMiddleware",
  wrapToolCall: async (request, handler) => {
    try {
      return await handler(request);
    } catch (e) {
      // 返回自定义的错误消息给LLM
      return new ToolMessage({
        content: `Tool error: Please check your input and try again. (${e})`,
        tool_call_id: request.toolCall.id || '',
      });
    }
  },
});



function getModelNameFromMessage(message: unknown): string {
  const meta = (message as any)?.response_metadata ?? (message as any)?.responseMetadata;
  return String(meta?.model_name ?? meta?.model ?? meta?.modelName ?? "");
}

/** 1.测试动态模型选择 */
async function testDynamicModelSelection() {
  const checkpointer = new MemorySaver();

  const agent = createAgent({
    model: dsModel,
    checkpointer,
    contextSchema,
    middleware: [dynamicModelMiddleware],
  });

  const config = { configurable: { thread_id: "1" }, context: { userId: "1" } };

  const r1 = await agent.invoke(
    { messages: [{ role: "user", content: "1.9 和1.11 哪个数字大？" }] },
    config,
  );
  const ai1 = r1.messages.at(-1);
  console.log("响应内容：\n", ai1?.content);
  console.log("调用模型：\n", getModelNameFromMessage(ai1));
  /*
  调用模型：
  deepseek-ai/DeepSeek-V3.2-Exp
  */

  const r2 = await agent.invoke(
    {
      messages: [
        {
          role: "user",
          content: "请用langchain 1.x 设计一个简单的问答系统，用户可以向系统咨询某地的天气信息，包括天气工具调用。",
        },
      ],
    },
    config,
  );
  const ai2 = r2.messages.at(-1);
  console.log("响应内容：\n", ai2?.content);
  console.log("调用模型：\n", getModelNameFromMessage(ai2));

  /*
  调用模型：
  Pro/zai-org/GLM-4.7
  */
}

/** 2.测试工具：比较两个数字 */
async function testToolCompareTwoNumbers() {
  const checkpointer = new MemorySaver();

  const agent = createAgent({
    model: dsModel,
    checkpointer,
    tools: [compareTwoNumbers],
    contextSchema,
    middleware: [ handleToolErrorsMiddleware],
  });

  const r = await agent.invoke(
    { messages: [{ role: "user", content: "1.9 和 1.11 哪个数字大？" }] },
    { configurable: { thread_id: "1" }, context: { userId: "1" } },
  );
  const ai = r.messages.at(-1);
  console.log("响应内容：\n", ai?.content);
  console.log("调用模型：\n", getModelNameFromMessage(ai));


  /*
  响应内容：
  比较结果是 **1**，这意味着 1.9 > 1.11。

  所以，**1.9** 比 **1.11** 大。

  虽然1.11在小数点后有两位数字，但比较小数大小时是看整体数值：
  - 1.9 实际上是 1.90
  - 1.11 是 1.11
  - 1.90 > 1.11
  调用模型：
  deepseek-ai/DeepSeek-V3.2-Exp
  */
}

const compareResultSchema = z.object({
  num1: z.number().describe("第一个数字"),
  num2: z.number().describe("第二个数字"),
  result: z.number().int().describe("比较结果，1 表示 num1 大于 num2，-1 表示 num1 小于 num2，0 表示相等"),
});


/** 测试响应格式 */
async function testResponseFormat() {
  const checkpointer = new MemorySaver();

  const agent = createAgent({
    model: glmModel,
    checkpointer,
    tools: [compareTwoNumbers],
    contextSchema,
    responseFormat: compareResultSchema,
    middleware: [handleToolErrorsMiddleware],
  });

  const r = await agent.invoke(
    { messages: [{ role: "user", content: "1.9 和 1.11 哪个数字大？" }] },
    { configurable: { thread_id: "1" }, context: { userId: "1" } },
  );

  console.log(r.structuredResponse);
  /*
  {
    num1: 1.9,
    num2: 1.11,
    result: 1,
  }
  */
  console.log(r.messages.at(-1)?.content);
  /*
    Returning structured response: {"num1":1.9,"num2":1.11,"result":1}
  */
}



if (import.meta.main) {
  
  // await testDynamicModelSelection();
  // await testToolCompareTwoNumbers();
  await testResponseFormat();
}
