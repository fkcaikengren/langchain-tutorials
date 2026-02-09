import { settings } from "@/config";
import { MemorySaver } from "@langchain/langgraph";
import { ChatOpenAI } from "@langchain/openai";
import { createAgent } from "langchain";

const model = new ChatOpenAI({
  model: settings.ds_model,
  apiKey: settings.siliconflow_api_key,
  configuration: {
    baseURL: settings.siliconflow_base_url,
  },
  temperature: 0.5,
  maxTokens: 5000,
  timeout: 60_000,
});

async function testNoCheckpointer() {
  console.log("\n" + "=".repeat(50));
  console.log("测试 1: createAgent 不带 checkpointer (应该无记忆)");
  console.log("=".repeat(50));

  const agent = createAgent({ model });
  const config = { configurable: { thread_id: "1" } };

  console.log("\n【步骤 1】\n [用户]: 嗨！我叫 Bob。");
  const response1 = await agent.invoke(
    { messages: [{ role: "user", content: "嗨！我叫 Bob。" }] },
    config,
  );
  console.log(`[Agent]: ${response1.messages.at(-1)?.content ?? ""}`);

  console.log("\n【步骤 2】\n [用户]: 我叫什么名字？");
  const response2 = await agent.invoke(
    { messages: [{ role: "user", content: "我叫什么名字？" }] },
    config,
  );
  console.log(`[Agent]: ${response2.messages.at(-1)?.content ?? ""}`);
}

async function testWithCheckpointer() {
  console.log("\n" + "=".repeat(50));
  console.log("测试 2: createAgent 带 checkpointer (应该有记忆)");
  console.log("=".repeat(50));

  const checkpointer = new MemorySaver();
  const agent = createAgent({ model, checkpointer });
  const config = { configurable: { thread_id: "thread-1" } };

  console.log("\n【步骤 1】\n [用户]: 嗨！我叫 Alice。");
  const response1 = await agent.invoke(
    { messages: [{ role: "user", content: "嗨！我叫 Alice。" }] },
    config,
  );
  console.log(`[Agent]: ${response1.messages.at(-1)?.content ?? ""}`);

  console.log("\n【步骤 2】\n [用户]: 我叫什么名字？");
  const response2 = await agent.invoke(
    { messages: [{ role: "user", content: "我叫什么名字？" }] },
    config,
  );
  console.log(`[Agent]: ${response2.messages.at(-1)?.content ?? ""}`);
}

async function testCheckpointerThreadIsolation() {
  console.log("\n" + "=".repeat(50));
  console.log("测试 3: createAgent 线程隔离");
  console.log("=".repeat(50));

  const checkpointer = new MemorySaver();
  const agent = createAgent({ model, checkpointer });

  console.log("\n[线程 A] 用户: 嗨！我叫 Charlie。");
  await agent.invoke(
    { messages: [{ role: "user", content: "嗨！我叫 Charlie。" }] },
    { configurable: { thread_id: "thread-A" } },
  );

  console.log("\n[线程 B] 用户: 你好！我叫 疯狂踩坑人");
  await agent.invoke(
    { messages: [{ role: "user", content: "你好！我叫 疯狂踩坑人" }] },
    { configurable: { thread_id: "thread-B" } },
  );

  console.log("\n[线程 A] 用户: 我叫什么名字？");
  const responseA = await agent.invoke(
    { messages: [{ role: "user", content: "我叫什么名字？" }] },
    { configurable: { thread_id: "thread-A" } },
  );
  console.log(`[线程 A] Agent: ${responseA.messages.at(-1)?.content ?? ""}`);

  console.log("\n[线程 B] 用户: 我叫什么名字？");
  const responseB = await agent.invoke(
    { messages: [{ role: "user", content: "我叫什么名字？" }] },
    { configurable: { thread_id: "thread-B" } },
  );
  console.log(`[线程 B] Agent: ${responseB.messages.at(-1)?.content ?? ""}`);
}

async function testCheckpoints() {
  console.log("\n" + "=".repeat(50));
  console.log("测试 4: 检查 checkpointer 保存的 checkpoint");
  console.log("=".repeat(50));

  const checkpointer = new MemorySaver();
  const agent = createAgent({ model, checkpointer });
  const config = { configurable: { thread_id: "thread-1" } };

  console.log("\n[用户]: 嗨！我叫 疯狂踩坑人。");
  const response = await agent.invoke(
    { messages: [{ role: "user", content: "嗨！我叫 疯狂踩坑人。" }] },
    config,
  );
  console.log(`[Agent]: ${response.messages.at(-1)?.content ?? ""}`);

  const checkpoints: unknown[] = [];
  for await (const checkpoint of checkpointer.list({ configurable: { thread_id: "thread-1" } })) {
    checkpoints.push(checkpoint);
  }
  console.log("checkpoint 数量：", checkpoints.length);
  for (const c of checkpoints) {
    console.log(c, "\n");
  }
}

if (import.meta.main) {
  // await testNoCheckpointer();
  // await testWithCheckpointer();
  // await testCheckpointerThreadIsolation();
  await testCheckpoints();
}

