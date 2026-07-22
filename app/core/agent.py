import os

# from langchain_community.chat_models import ChatOpenAI
from langchain_openai import ChatOpenAI
from langchain.tools import Tool
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from app.tools import knowledge_search, web_search, calculator
from dotenv import load_dotenv
from pydantic import SecretStr
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import InMemorySaver
from app.memory.user_profile import UserProfileMemory

load_dotenv()
_agent_executor = None
_checkpointer = InMemorySaver()
_profile_memory = UserProfileMemory()


def get_agent(user_id: str = "default"):
    global _agent_executor
    if _agent_executor is None:
        _agent_executor = create_agent(user_id)
    return _agent_executor


def create_agent(user_id: str):
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
    # ✅ 获取用户画像，注入 System Prompt
    user_context = _profile_memory.get_context_prompt(user_id)
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                f"""你是一个智能助手。当用户提出问题时，**必须**考虑是否需要使用工具。
                {user_context if user_context else "暂无用户信息，请通过对话了解用户。"}
                1. 用户没有明确询问公司信息时，不要主动提及公司
                2. 只回答用户直接问的问题，不要过度延伸
                3. 记住用户说的个人信息，但不要自己发挥
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
            MessagesPlaceholder(variable_name="messages"),
        ]
    )
    agent = create_react_agent(
        model=llm,
        tools=tools,
        prompt=prompt,
        checkpointer=_checkpointer,
    )
    return agent
