import { settings } from "@/config";
import { StringOutputParser } from "@langchain/core/output_parsers";
import { HumanMessage } from "@langchain/core/messages";
import { ChatOpenAI } from "@langchain/openai";

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

async function testInvoke() {
  /*
  invoke 调用
  */
  // const res = await model.invoke("Translate 'I love programming' into Chinese.");
  // 等价于
  // const res = await model.invoke([{ role: "user", content: "Translate 'I love programming' into Chinese." }]);
  // 等价于
  const res = await model.invoke([new HumanMessage({ content: "Translate 'I love programming' into Chinese." })]);
  console.log(res);
  /*
  返回：AIMessage
  {
      "content": "我喜欢编程。",
      "response_metadata": {...},
      "id": "...",
      ...
  }
  */

  // 通过管道pipe 链接模型和解析器，返回一个增强的链。 新的链调用模型的结果 会从 AIMessage 转换为字符串
  const chain = model.pipe(new StringOutputParser()); // 这就是 LCEL (LangChain Expression Language)
  const ans = await chain.invoke("Translate 'I love programming' into Chinese.");
  console.log(ans);
  /*
  输出：我喜欢编程。
  */
}

async function testStream() {
  /*
  Stream 调用 (流式输出)
  */
  const chain = model.pipe(new StringOutputParser());
  const messages: [string, string][] = [
    [
      "system",
      "You are a helpful translator. Translate the user sentence to Chinese.",
    ],
    ["human", "I love programming."],
  ];

  const stream = await chain.stream(messages); // 返回一个异步迭代器，可以用for await 遍历
  for await (const chunk of stream) {
    console.log(chunk);
  }
  /*
  输出：
      我很
      喜欢
      编程。
  */
}

async function testBatch() {
  /*
  # Batch 调用 (并行执行多个请求)
  数组的每一项是一个请求, 请求是并行的
  */
  const chain = model.pipe(new StringOutputParser());
  const batches = await chain.batch([
    [
      {
        role: "system",
        content: "You are a helpful translator. Translate the sentence to Chinese.",
      },
      { role: "human", content: "I love programming." },
    ],
    [{ role: "human", content: "100字内，介绍下langchain。" }],
  ]);

  for (let i = 0; i < batches.length; i += 1) {
    console.log(`Result ${i + 1}:\n${batches[i]}\n`);
  }

  /*
  输出：
      Result 1:
      我非常热爱编程。

      Result 2:
      LangChain是一个开源框架，旨在简化基于大型语言模型（LLM）的应用程序开发。它通过提供模块化组件和工具链，帮助开发者轻松连接LLM与外部数据源、API或计算资源，实现数据感知和代理式交互应用。核心功能包括Prompt模板化、记忆管理、链式调用及多工具集成，大幅提升开发效率。
  */


  // 控制并发
  // const batches = await chain.batch(
  //   [
  //     '翻译"I love programming."成中文',
  //     "100字内，介绍下框架langchain。",
  //     "100字内，介绍下js运行时 bun。",
  //   ],
  //   {
  //     maxConcurrency: 2,  // 限制并发数为2
  //   }
  // )
  // console.log(batches);
}

if (import.meta.main) {
  // await testInvoke();
  // await testStream();
  await testBatch();
}
