## 目标
- 参考并对齐 Python 版本：`langchain_py/app/1/quickstart.py`、`lcel.py`、`invocation.py`。
- 在 `langchain_ts/app/1/` 下分别实现 `quickstart.ts`、`lcel.ts`、`invocation.ts`。
- 保留原有注释（包括注释掉的代码也照搬保留），并遵循项目规范：使用 Bun 运行。

## 实现要点（按文件）
### quickstart.ts
- 复刻 Python 中的 6 步流程与对应注释：提示词、工具、模型、结构化输出、memory、创建并调用 agent。
- 使用 TS 侧的 `createAgent`/`tool`/`toolStrategy`（来自 `langchain`）来实现与 Python `create_agent` + `ToolStrategy` 等价的结构化输出。
- 模型使用 `@langchain/openai` 的 `ChatOpenAI`，并从 `@/config` 读取 `settings.glm_model / settings.siliconflow_*` 来配置 OpenAI-兼容第三方接口。
- Memory 使用 `@langchain/langgraph` 的 `MemorySaver` 作为 checkpointer，并通过 `configurable.thread_id` 实现同一会话复用。
- 工具 `get_user_location` 通过工具回调的第二个参数 `RunnableConfig` 读取 `config.configurable.user_id`，以达到 Python `ToolRuntime[Context].context.user_id` 的效果。
- 调用两次 `agent.invoke`：第一次问天气打印 `structuredResponse`，第二次在同一 `thread_id` 下继续对话并再次打印。

### invocation.ts
- 复刻 `invoke / stream / batch` 三个示例函数与对应注释。
- 仍复用同一 `ChatOpenAI` 配置。
- 用 `StringOutputParser`（`@langchain/core/output_parsers`）实现与 Python `StrOutputParser` 等价的管道：`model.pipe(parser)`。
- `stream` 使用 `for await ... of await chain.stream(...)` 输出分块。
- `batch` 使用 `await chain.batch([...])` 并逐个打印结果。
- 保留 Python 中的“等价写法”注释（用 TS/JS 能表达的形式写出来）。

### lcel.ts
- 复刻 5 个测试函数：
  - `test_prompt_template`：`HumanMessagePromptTemplate / ChatPromptTemplate` 的 invoke/format 行为与打印。
  - `test_output_parser`：`StringOutputParser` 的简单调用。
  - `test_lcel`：`prompt | model | parser` 的等价写法（TS 使用 `.pipe` 形成链）。
  - `test_runnable_sequence`：`RunnableLambda` + `RunnableSequence`（或 `.pipe`）完成分类器示例。
  - `test_runnable_branch`：用 `model.withStructuredOutput(zodSchema)` 得到结构化分类结果，再用 `RunnablePassthrough.assign` + `RunnableBranch.from` 做三分支路由，并保留 debug 打印。
- 保留 Python 中 `__main__` 的调用开关；TS 中用 `if (import.meta.main) { ... }`。

## 验证方式（实现后执行）
- 逐个用 Bun 运行三个脚本，确认可以正常发起模型请求并打印示例输出（需要已正确配置环境变量）。
- 重点验证：
  - `quickstart.ts` 同一 `thread_id` 的第二次调用能延续上下文；
  - `invocation.ts` 的 `stream/batch` 可运行；
  - `lcel.ts` 的 runnable branch 能根据结构化分类正确路由。

## 变更范围
- 只修改/填充以下 3 个空文件：
  - `langchain_ts/app/1/quickstart.ts`
  - `langchain_ts/app/1/lcel.ts`
  - `langchain_ts/app/1/invocation.ts`
- 不改动其余文件（如 `index.ts`）除非验证需要最小化入口调整。