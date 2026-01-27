
# langchain_py项目


使用`uv`做包管理器和管理项目的python环境。 

版本建议：   
python==3.12

### 安装依赖和配置
```bash
# 初始化环境和安装依赖
uv sync
```

复制`.env.example`文件为`.env`文件，并配置自己的API KEY。


### 运行示例

运行`app/1/quickstart.py`文件
```bash
cd langchain_py 
uv run -m app.1.quickstart
```
