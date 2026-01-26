

# langchain_py项目规范
python==3.12
使用uv做包管理器和项目管理。
通过`uv run -m app.xx`来运行app目录下的xx.py文件

app/agent.py 使用 qwen3_32b_model 判断问题复杂度：simple 走 ds_model，complex 走 glm_model
运行示例：`uv run python -m app.agent`
