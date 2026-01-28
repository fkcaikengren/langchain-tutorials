import { settings } from "@/config";
import { ChatOpenAI } from "@langchain/openai";
import { ChatDeepSeek } from "@langchain/deepseek";
import { initChatModel } from "langchain";


async function testDeepSeekModel() {
  // 使用 ChatDeepSeek
//   const model = new ChatDeepSeek({
//     model: "deepseek-chat",
//     apiKey: settings.deepseek_api_key,
//     temperature: 0.9,
//     maxTokens: 1000,
//     timeout: 60000, // TS 中 timeout 通常是毫秒
//   });

    const model = await initChatModel("deepseek-chat", {
    modelProvider: "deepseek",
    apiKey: settings.deepseek_api_key,
    temperature: 0.9,
    maxTokens: 1000,
    timeout: 60000, // TS 中 timeout 通常是毫秒
  });

  const response = await model.invoke("你好");
  console.log(response.content);
}

/**
 * 测试第三方模型，只要符合 openai 规范的 api，都可以用 ChatOpenAI 来创建该模型的实例
 */
async function testThirdPartModel() {
  const model = new ChatOpenAI({
    model: settings.ds_model,
    apiKey: settings.siliconflow_api_key,
    configuration: {
      baseURL: settings.siliconflow_base_url,
    },
    temperature: 0.9,
    maxTokens: 1000,
    timeout: 60000,
  });

  const response = await model.invoke("你好");
  console.log(response.content);
}

if (import.meta.main) {
//   testDeepSeekModel()
  testThirdPartModel()
}
