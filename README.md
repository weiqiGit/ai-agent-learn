AI Agent 服务

面向企业内部的 AI Agent 后端服务，支持 RAG 知识库问答、多工具调用、人工审批（HITL）和三层记忆管理。

功能特性

- RAG 知识库问答：上传 PDF/TXT 文档，自动切片、向量化、检索，回答时标注来源
- 自然语言转 SQL：用户用自然语言描述需求，LLM 自动生成 SQL，经人工确认后查询数据库并给出分析结论
- 多工具调用：集成知识库检索、计算器、联网搜索、SQL 查询 4 个工具，Agent 自主选择
- 人工审批（HITL）：SQL 生成后暂停，前端弹窗展示 SQL，用户确认后才执行
- 三层记忆：短期记忆（会话级）、标签记忆（结构化）、长期记忆（向量化）
- 流式输出：支持打字机效果、工具调用状态可视化
- 结构化日志：每次请求记录用户、问题、工具调用、耗时等字段
- 全局异常处理：统一捕获、统一日志、统一返回格式
- 工具重试机制：指数退避 + 随机抖动

技术栈
前端： Next.js、TypeScript 
后端： Python、FastAPI
Agent 编排： LangGraph、LangChain 
大模型： DeepSeek
Embedding： 智谱 Embedding 
向量存储： Chroma 
关系数据库： SQLite 
联网搜索： Tavily 

快速开始

后端（访问 http://localhost:8000/docs 查看接口文档。）

```bash
pip install -r requirements.txt
cp .env.example .env  # 填入 API Key
python scripts/init_db.py
uvicorn app.main:app --reload

项目结构

app/
├── api/          # 接口层
├── core/         # Agent 编排、RAG 引擎、异常处理
├── memory/       # 三层记忆
├── tools/        # 工具（知识库、搜索、计算器、SQL）
├── utils/        # 日志、重试
└── main.py       # FastAPI 入口


