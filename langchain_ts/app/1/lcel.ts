import { settings } from "@/config";
import {
  AIMessagePromptTemplate,
  ChatPromptTemplate,
  HumanMessagePromptTemplate,
  SystemMessagePromptTemplate,
} from "@langchain/core/prompts";
import {
  RunnableBranch,
  RunnableLambda,
  RunnablePassthrough,
  RunnableSequence,
} from "@langchain/core/runnables";
import { StringOutputParser } from "@langchain/core/output_parsers";
import { AIMessage, HumanMessage, SystemMessage } from "@langchain/core/messages";
import { ChatOpenAI } from "@langchain/openai";
import { z } from "zod";

const model = new ChatOpenAI({
  model: settings.glm_model,
  apiKey: settings.siliconflow_api_key,
  configuration: {
    baseURL: settings.siliconflow_base_url,
  },
  temperature: 0.9,
  maxTokens: 3000,
  timeout: 60000,
});

async function testPromptTemplate() {
  /*
  测试提示模板:
      SystemMessagePromptTemplate
      HumanMessagePromptTemplate
      AIMessagePromptTemplate
      ChatPromptTemplate
  */
  const humanPrompt = HumanMessagePromptTemplate.fromTemplate(
    "你是一个专业的翻译。请将以下文本从英文翻译为中文：\n {input}",
  );
  const humanMessage = await humanPrompt.format({ input: "I love programming." });
  console.log(
    "h_message 是 HumanMessage 类型吗？",
    humanMessage instanceof HumanMessage,
  ); // true
  console.log(humanMessage);
  // 输出
  // HumanMessage(content='你是一个专业的翻译。请将以下文本从英文翻译为中文：\n I love programming.', additional_kwargs={}, response_metadata={})

  const chatPrompt = ChatPromptTemplate.fromMessages([
    ["system", "你是一个{role}。"],
    ["human", "请将以下文本从英文翻译为中文：\n {userInput}"],
  ]);
  // ChatPromptTemplate 是runnable, 调用invoke 。 返回 Message列表
  const chatMessages = await chatPrompt.invoke({
    role: "专业的翻译",
    userInput: "I love programming.",
  });
  console.log(chatMessages);
  // 输出：
  /*
  ChatPromptValue {
    lc_serializable: true,
    ...
    messages: [
      SystemMessage {
        "content": "你是一个专业的翻译。",
        "additional_kwargs": {},
        "response_metadata": {}
      }, HumanMessage {
        "content": "请将以下文本从英文翻译为中文：\n I love programming.",
        "additional_kwargs": {},
        "response_metadata": {}
      }
    ],
  }
  */

  // ！注意：format 方法返回的是一个字符串
  const chatPromptStr = await chatPrompt.format({
    role: "专业的翻译",
    userInput: "I love programming.",
  });
  console.log(typeof chatPromptStr); // string
  console.log(chatPromptStr);
  /*
    System: 你是一个专业的翻译。
    Human: 请将以下文本从英文翻译为中文：
     I love programming.
  */
}

async function testOutputParser() {
  /*
  测试输出解析器:StringOutputParser
  */
  const parser = new StringOutputParser();
  const ans = await parser.invoke("翻译 'I love programming' 成中文。");
  console.log(ans);
  // 输出：
  // 翻译 'I love programming' 成中文。
}

async function testLcel() {
  /*
  测试 LCEL (LangChain Expression Language)
  */
  const chatPrompt = ChatPromptTemplate.fromMessages([
    ["system", "你是一个{role}。"],
    ["human", "请将以下文本从英文翻译为中文：\n {userInput}"],
  ]);

  const chain = chatPrompt.pipe(model).pipe(new StringOutputParser());
  const ans = await chain.invoke({
    role: "专业的翻译",
    userInput: "I love programming.",
  });
  console.log(ans);
  // 输出：
  // 我热爱编程。
}

async function testRunnableSequence() {
  const prompt = ChatPromptTemplate.fromMessages([
    ["system", "你是一个{type}，请根据用户的提问进行回答"],
    ["system", "{instruction}"],
    ["human", "{question}"],
  ]);

  const classifier = (input: { question: string; instruction: string }) => {
    const { question, instruction } = input;
    const typeValue = question.includes("科普") ? "科普专家" : "智能助手";
    return {
      type: typeValue,
      question,
      instruction,
    };
  };

  // RunnableLambda.from 将classifier包装成一个Runnable，使它可以在LCEL中使用
  // const chain = RunnableLambda.from(classifier).pipe(prompt).pipe(model).pipe(new StringOutputParser());
  // 等价于
  // const chain = RunnableSequence.from([
  //   RunnableLambda.from(classifier),
  //   prompt,
  //   model,
  //   new StringOutputParser(),
  // ]);
  // 等价于  
  const chain = RunnableSequence.from([
    classifier,//直接传入函数， RunnableSequence内会被自动包装成Runnable
    prompt,
    model,
    new StringOutputParser(),
  ]);

  const ans = await chain.invoke({
    question: "科普，鲸鱼是哺乳动物么？只需要回答是或不是",
    instruction: "用中文回答",
  });
  console.log(ans);
  // 是
}

async function testRunnableBranch() {
  const classifyResultSchema = z.object({
    type: z.enum(["科普", "编程", "其他"]),
  });

  const structuredModel = model.withStructuredOutput(classifyResultSchema);

  const sciencePrompt = ChatPromptTemplate.fromMessages([
    ["system", "你是科普专家，通俗准确、简洁回答。"],
    ["human", "{question}"],
  ]);
  const scienceExpert = RunnableSequence.from([
    sciencePrompt,
    model,
    new StringOutputParser(),
  ]);

  const codePrompt = ChatPromptTemplate.fromMessages([
    ["system", "你是编程专家，提供代码或技术解答。"],
    ["human", "{question}"],
  ]);
  const codeExpert = RunnableSequence.from([codePrompt, model, new StringOutputParser()]);

  const generalPrompt = ChatPromptTemplate.fromMessages([
    ["system", "你是智能助手，简洁中文回答。"],
    ["human", "{question}"],
  ]);
  const generalExpert = RunnableSequence.from([
    generalPrompt,
    model,
    new StringOutputParser(),
  ]);

  const classifier = RunnablePassthrough.assign({
    // classifyResult 会 赋值给 input["classifyResult"]
    classifyResult: async (input: { question: string }) => {
      return structuredModel.invoke(
        `请判断以下问题的类型，并只返回JSON：{"type": "科普"|"编程"|"其他"}。问题：${input.question}`,
      );
    },
  });

  const debugRunnable = RunnableLambda.from((input: unknown) => {
    console.log("中间结果:", input);
    return input;
  });

  // RunnableBranch 将从三个分支中选取一个，根据lambda函数的返回值判断选择哪一个。
  const branch = RunnableSequence.from([
    classifier,
    debugRunnable,
    RunnableBranch.from([
      [
        (input: { classifyResult?: { type?: string } }) =>
          input.classifyResult?.type === "科普",
        scienceExpert,
      ],
      [
        (input: { classifyResult?: { type?: string } }) =>
          input.classifyResult?.type === "编程",
        codeExpert,
      ],
      generalExpert,
    ]),
  ]);

  const result = await branch.invoke({ question: "简单回答下，鲸鱼是哺乳动物吗？" });
  console.log(result);
  /*
    中间结果: {
      question: "简单回答下，鲸鱼是哺乳动物吗？",
      classifyResult: {
        type: "科普",
      },
    }

    是的，鲸鱼是哺乳动物。
  */
}

if (import.meta.main) {
  // await testPromptTemplate();
  // await testOutputParser();
  // await testLcel();
  // await testRunnableSequence();
  await testRunnableBranch();
}
