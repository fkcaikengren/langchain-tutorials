import { settings } from "@/config";
import { HumanMessage } from "@langchain/core/messages";
import type { ToolCall } from "@langchain/core/messages/tool";
import { ToolMessage } from "@langchain/core/messages/tool";
import { ChatOpenAI } from "@langchain/openai";
import { createAgent, tool } from "langchain";
import type { ToolRuntime } from "langchain"
import { z } from "zod";


const ToolContext = z.object({
  userId: z.string().describe("用户ID"),
})

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




// 定义工具
const getReviews = tool(
  ({ positive }: { positive: boolean }) => {

    const positiveReviews = [
      "原来两三岁的小孩也可以不扯女孩裙子啊；原来不整屎尿屁也可以做出让全场大笑的效果啊；原来女角色也可以不穿超短裙高开叉高跟鞋啊；原来男师父女徒弟也可以不暧昧纯师徒情啊；原来一个动画片里正派之间也可以有不同的价值观啊；原来不喊口号不献祭亲朋好友父老乡亲也能表达反战的思想啊。罗小黑你还是太超前了。",
      "瑕不掩瑜。非常好的一点是，一点儿爹味都没有，不judge任何人（妖精），没有任何人（妖精）需要被打败或悔过。这在中国的大型说教重灾区———国漫中已是十分可贵。",
      "“无限虽然爱装逼，但是他没有跟鹿野搞花千骨，此乃一胜；没有跟罗小黑搞黑猫和他的蓝发师尊，此乃二胜；没有和哪吒搞男同，此乃三胜”",
      "我宣布鹿野是我唯一的姐！太帅了！！！工装裤配T恤，低马尾，非传统女性角色，太帅了5555555希望越来越强，早日拳打无限脚踢各大长老！！！ 以及，真是好多场经费爆炸的打斗啊",
    ];
    const negativeReviews = [
      "呃…片方到底懂不懂自己的IP魅力在哪啊！搞什么武器、战争的宏大场面啊，又搞不明白，妥妥露怯！整个剧情就是，稀碎…",
    ];
    return positive ? positiveReviews : negativeReviews;
  },
  {
    name: "get_reviews",
    description: "获取罗小黑电影评论列表",
    schema: z.object({
      positive: z.boolean().describe("是否获取正面评论, true 表示正面，false 表示负面"),
    }),
  },
);


// 定义工具（测试运行时）
const getReviewsWithRuntime = tool(
  async ({ positive }: { positive: boolean }, config: ToolRuntime<any, typeof ToolContext>) => {
    console.log("参数positive", positive);
    console.log("1.查看对话的状态");
    console.log(config.state.messages);
    // [HumanMessage(content='请分析罗小黑电影的正面评论原因？'), AIMessage(content='我来帮您获取罗小黑电影的正面评论并分析其中的原因。'), ...]

    console.log("2.通过context 可以查询user的个人信息");
    console.log(config.context)
    // {
    //   userId: "user123",
    // }

    console.log("3.在长任务中，反馈进度，通常配合langgraph使用");
    const writer = config.writer;
    if(writer){
      writer({ status: "starting", message: "正在处理查询" });
      writer({ status: "progress", message: "完成50%" });
    }

    const positiveReviews = [
      "原来两三岁的小孩也可以不扯女孩裙子啊；原来不整屎尿屁也可以做出让全场大笑的效果啊；原来女角色也可以不穿超短裙高开叉高跟鞋啊；原来男师父女徒弟也可以不暧昧纯师徒情啊；原来一个动画片里正派之间也可以有不同的价值观啊；原来不喊口号不献祭亲朋好友父老乡亲也能表达反战的思想啊。罗小黑你还是太超前了。",
      "瑕不掩瑜。非常好的一点是，一点儿爹味都没有，不judge任何人（妖精），没有任何人（妖精）需要被打败或悔过。这在中国的大型说教重灾区———国漫中已是十分可贵。",
      "“无限虽然爱装逼，但是他没有跟鹿野搞花千骨，此乃一胜；没有跟罗小黑搞黑猫和他的蓝发师尊，此乃二胜；没有和哪吒搞男同，此乃三胜”",
      "我宣布鹿野是我唯一的姐！太帅了！！！工装裤配T恤，低马尾，非传统女性角色，太帅了5555555希望越来越强，早日拳打无限脚踢各大长老！！！ 以及，真是好多场经费爆炸的打斗啊",
    ];
    const negativeReviews = [
      "呃…片方到底懂不懂自己的IP魅力在哪啊！搞什么武器、战争的宏大场面啊，又搞不明白，妥妥露怯！整个剧情就是，稀碎…",
    ];
    return positive ? positiveReviews : negativeReviews;
  },
  {
    name: "get_reviews_with_runtime",
    description: "获取罗小黑电影评论列表（演示工具运行时读取 config）",
    schema: z.object({
      positive: z.boolean().describe("是否获取正面评论, true 表示正面，false 表示负面"),
    }),
  },
);




/**
 * 测试工具调用
 */
async function testToolCalling() {
  
  // const reviews = await getReviews.invoke({ positive: true });
  // console.log(reviews); // 输出reviews数组: ["原来两三岁...",...]

  const tools = [getReviews];
  
  const modelWithTools = model.bindTools(tools);

  const response = await modelWithTools.invoke("请分析罗小黑电影的负面评论原因？");
  const toolCalls = response.tool_calls ?? [];
  for (const toolCall of toolCalls) {
    console.log(`Tool: ${toolCall.name}`);
    console.log(`Args: ${JSON.stringify(toolCall.args)}`);
  }
  /* 输出
    Tool: get_reviews
    Args: {"positive":false}
  */
}

/**
 * 测试工具调用2（测试多轮对话）
 */
async function testToolCalling2() {
  
  const tools = [getReviews];
  const toolByName = Object.fromEntries(tools.map((t) => [t.name, t])) as Record<
    string,
    (typeof tools)[number]
  >;
  const modelWithTools = model.bindTools(tools);
  

  const prompt = "请分析罗小黑电影的正面评论原因？";
  const response = await modelWithTools.invoke(prompt);

  const toolMessages: ToolMessage[] = [];
  const toolCalls =  response.tool_calls ?? [];
  for (const toolCall of toolCalls) {
    console.log(`Tool: ${toolCall.name}`);
    console.log(`Args: ${JSON.stringify(toolCall.args)}`);
    const tool = toolByName[toolCall.name];
    if (!tool) {
      throw new Error(`Unknown tool: ${toolCall.name}`);
    }
    // @ts-expect-error
    const reviews = await tool.invoke(toolCall.args);
    toolMessages.push(
      new ToolMessage({
        content: JSON.stringify(reviews),
        tool_call_id: toolCall.id ?? "",
      }),
    );
  }

  const finalResponse = await modelWithTools.invoke([
    new HumanMessage({ content: prompt }),
    response,
    ...toolMessages,
  ]);
  console.log(finalResponse.content);
  /*输出：
    Tool: get_reviews
    Args: {"positive":true}
    基于获取到的正面评论，我来为您分析罗小黑电影受欢迎的主要原因：
    ## 罗小黑电影正面评论分析
    ### 1. **突破传统套路，创新性强**
    观众普遍认为这部电影"太超前了"，主要体现在：
    - **儿童角色塑造**：两三岁的小孩角色不惹麻烦，有良好行为
    - **幽默表现手法**：不依赖粗俗的屎尿屁笑话也能制造全场爆笑效果
        ...
  */
}


/**
 * 测试工具调用3 (仅仅把问题改了，测试并行调用工具)
 */
async function testToolCalling3() {
  
  const tools = [getReviews];
  const toolByName = Object.fromEntries(tools.map((t) => [t.name, t])) as Record<
    string,
    (typeof tools)[number]
  >;
  const modelWithTools = model.bindTools(tools);

  const prompt = "请分析罗小黑电影的正面评论原因和负面评论原因？";
  const response = await modelWithTools.invoke(prompt);

  const toolMessages: ToolMessage[] = [];
  const toolCalls =  response.tool_calls ?? [];
  for (const toolCall of toolCalls) {
    console.log(`Tool: ${toolCall.name}`);
    console.log(`Args: ${JSON.stringify(toolCall.args)}`);
    const tool = toolByName[toolCall.name];
    if (!tool) {
      throw new Error(`Unknown tool: ${toolCall.name}`);
    }
    // @ts-expect-error
    const reviews = await tool.invoke(toolCall.args);
    toolMessages.push(
      new ToolMessage({
        content: JSON.stringify(reviews),
        tool_call_id: toolCall.id ?? "",
      }),
    );
  }

  const finalResponse = await modelWithTools.invoke([
    new HumanMessage({ content: prompt }),
    response,
    ...toolMessages,
  ]);
  console.log(finalResponse.content);
  /*输出：
      Tool: get_reviews
      Args: {"positive":true}
      Tool: get_reviews
      Args: {"positive":false}
      ...
  */
}

/**
 * 测试工具运行时
 */
async function testToolRuntime() {
  const agent = createAgent({
    model,
    tools: [getReviewsWithRuntime],
    systemPrompt: "你是影评分析助手，请在需要时调用工具获取评论再进行分析。",
    contextSchema: ToolContext
  });

  await agent.invoke(
    {
      messages: [
        {
          role: "user",
          content: "请分析罗小黑电影的负面评论原因？",
        },
      ],
    },
    { context: { userId: "user123" } },
  );
}


if (import.meta.main) {
  // await testToolCalling()
  // await testToolCalling2()
  // await testToolCalling3();
  await testToolRuntime();
}
