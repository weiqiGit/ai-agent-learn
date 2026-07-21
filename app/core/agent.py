import os

# from langchain_community.chat_models import ChatOpenAI
from langchain_openai import ChatOpenAI
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain.tools import Tool
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from app.tools import knowledge_search, web_search, calculator
from dotenv import load_dotenv
from pydantic import SecretStr
from langchain.memory import ConversationSummaryMemory

load_dotenv()
_agent_executor = None


def get_agent():
    global _agent_executor
    if _agent_executor is None:
        _agent_executor = create_agent()
    return _agent_executor


def create_agent():
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise ValueError("请设置环境变量 DEEPSEEK_API_KEY")
    llm = ChatOpenAI(
        model="deepseek-chat",
        api_key=SecretStr(api_key),
        base_url="https://api.deepseek.com/v1",
        temperature=0.3,
    )

    tools = [
        Tool(
            name="knowledge_search",  # ✅ 英文
            func=knowledge_search,
            description="查询公司内部文档、制度、流程、产品手册等。当用户问关于公司内部的事时使用。",
        ),
        Tool(
            name="web_search",  # ✅ 英文
            func=web_search,
            description="搜索互联网上的最新信息、新闻、实时数据。当需要查询外部信息时使用。",
        ),
        Tool(
            name="calculator",  # ✅ 英文
            func=calculator,
            description="执行数学计算，如加减乘除、百分比等。当用户需要计算时使用。",
        ),
    ]

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """你是一个智能助手。当用户提出问题时，**必须**考虑是否需要使用工具。

工具列表：
1. knowledge_search：查询公司内部文档
2. web_search：搜索互联网
3. calculator：执行数学计算（加减乘除等）

**重要规则：**
- 用户问数学计算时，**必须**使用 calculator 工具，不要自己计算
- 用户问公司内部问题，使用 knowledge_search
- 用户问外部信息，使用 web_search
- 问候类问题可以不使用工具直接回答

请根据用户问题，选择合适的工具。
""",
            ),
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}"),
            # agent_scratchpad是多步推理的草稿纸
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )
    # ConversationBufferMemory-基础记忆，ConversationBufferWindowMemory-窗口记忆，ConversationSummaryMemory摘要记忆
    memory = ConversationSummaryMemory(
        memory_key="chat_history", return_messages=True, llm=llm
    )
    agent = create_tool_calling_agent(llm, tools, prompt)
    executor = AgentExecutor(
        agent=agent,
        tools=tools,
        memory=memory,
        verbose=True,
        handle_parsing_errors=True,
        return_intermediate_steps=True,
        max_iterations=5,
    )
    return executor
