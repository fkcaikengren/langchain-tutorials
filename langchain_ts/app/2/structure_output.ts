import { settings } from "@/config";
import { ChatOpenAI } from "@langchain/openai";
import { z } from "zod";
import { CommaSeparatedListOutputParser, StructuredOutputParser } from "@langchain/core/output_parsers";
import { createAgent, providerStrategy, toolStrategy } from "langchain";

const model = new ChatOpenAI({
  model: settings.glm_model,
  apiKey: settings.siliconflow_api_key,
  configuration: {
    baseURL: settings.siliconflow_base_url,
  },
  temperature: 0.9,
  maxTokens: 5000,
  timeout: 60_000,
});

const MovieSchema = z
  .object({
    title: z.string().describe("电影名称"),
    year: z.number().int().describe("电影上映时间"),
    director: z.string().describe("电影的导演"),
    rating: z.number().describe("电影的豆瓣评分"),
  })
  .describe("电影的相关信息");

const DevProcessListSchema = z.array(z.string()).describe("按顺序的软件开发流程字符串数组");

/**
 * 测试结构化输出
 */
async function testStructureClass() {
  const modelWithStructure = model.withStructuredOutput(MovieSchema);
  const response = await modelWithStructure.invoke([
    {
      role: "user",
      content: "介绍下电影《罗小黑战记2》，获取title、year、director、rating信息",
    },
  ]);
  console.log(typeof response); // object
  console.log(response);
  /*
  {
    title: "罗小黑战记2：在那个夏日",
    year: 2023,
    director: "不亦舟",
    rating: 8.8,
  }
  */
}

/**
 * 测试结构化输出（数组）
 */
async function testStructureList() {
  // 使用LLM provider API 强制结构化输出
  const modelWithStructure = model.withStructuredOutput(DevProcessListSchema);

  // 使用langchain自己的解析器 获取结构化输出（可靠性一般）
  // const outputParser = StructuredOutputParser.fromZodSchema(DevProcessListSchema);
  // const modelWithStructure = model.pipe(outputParser);

  const response = await modelWithStructure.invoke([
    {
      role: "user",
      content: "软件开发的流程是？请给我一个有顺序的字符串数组",
    },
  ]);
  console.log(Array.isArray(response)); //true
  console.log(response); 
  // [ "需求分析", "系统设计", "编码实现", "软件测试", "部署上线", "运维监控", "版本迭代" ]

  // P.S. 使用CommaSeparatedListOutputParser （复杂格式准确率相对不高，但更快更节省token）
  // const modelWithStructure = model.pipe(new CommaSeparatedListOutputParser());
  // const response = await modelWithStructure.invoke([
  //   {
  //     role: "user",
  //     content: "软件开发的流程是？请给我一个有顺序的字符串列表",
  //   },
  // ]);
  // console.log(response);
  // // [ "以下是软件开发流程（SDLC）的标准顺序列表：\n\n1. 需求分析\n2. 系统设计\n3. 编码开发\n4. 系统测试\n5. 部署发布\n6. 维护运维" ]
}


async function testStrategy(){
  const ContactInfoSchema = z
    .object({
      name: z.string().describe("The name of the person"),
      email: z.string().describe("The email address of the person"),
      phone: z.string().describe("The phone number of the person"),
    })
    .describe("Contact information for a person.");

  const tools: never[] = [];
  const systemPrompt = "你是信息抽取助手。请根据用户输入抽取联系人信息。";

  const input = {
    messages: [
      {
        role: "user" as const,
        content:
          "从下面文本中提取联系人信息（name/email/phone）：张三，邮箱 zhangsan@example.com，电话 13800000000。",
      },
    ],
  };

  const agentAuto = createAgent({
    model,
    tools,
    systemPrompt,
    responseFormat: ContactInfoSchema,
  });

  const agentProvider = createAgent({
    model,
    tools,
    systemPrompt,
    responseFormat: providerStrategy(ContactInfoSchema),
  });

  const agentTool = createAgent({
    model,
    tools,
    systemPrompt,
    responseFormat: toolStrategy(ContactInfoSchema),
  });

  const autoResult = await agentAuto.invoke(input);
  console.log("auto", autoResult.structuredResponse);
  /*
  auto {
    name: "张三",
    email: "zhangsan@example.com",
    phone: "13800000000",
  }
  */
  try {
    const providerResult = await agentProvider.invoke(input);
    console.log("provider", providerResult.structuredResponse);
  } catch (err) {
    console.error("provider failed", err);
  }

  const toolResult = await agentTool.invoke(input);
  console.log("tool", toolResult.structuredResponse);
  /*
  tool {
    name: "张三",
    email: "zhangsan@example.com",
    phone: "13800000000",
  }
  */
}


if (import.meta.main) {
  // await testStructureClass();
  // await testStructureList();
  await testStrategy()
}
