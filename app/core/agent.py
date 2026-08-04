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


def get_agent(user_id: str = "default", memory_context: str = ""):
    global _agent_executor
    if _agent_executor is None:
        _agent_executor = create_agent(user_id, memory_context)
    return _agent_executor


def create_agent(user_id: str, memory_context: str = ""):
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
    # ✅ 获取结构化用户画像
    user_context = _profile_memory.get_context_prompt(user_id)

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                f"""你是一个智能助手。当用户提出问题时，**必须**考虑是否需要使用工具。
                【用户信息】
{user_context if user_context else "暂无用户信息"}
【用户记忆】
{memory_context if memory_context else "暂无用户记忆"}
                1. 用户没有明确询问公司信息时，不要主动提及公司
                2. 只回答用户直接问的问题，不要过度延伸
                3. 记住用户说的个人信息，但不要自己发挥
                工具列表：
                1. knowledge_search：查询公司内部文档
                2. web_search：搜索互联网
                3. calculator：执行数学计算（加减乘除等）

                **重要规则：**
                - 当用户提到公司相关术语（如工作时间、打卡、工资、报销、年假、制度等）时，**必须**调用 knowledge_search，不要分析用户意图，不要反问用户，直接查询
                - 用户问数学计算时，**必须**使用 calculator 工具，不要自己计算
                - 用户问公司内部问题，使用 knowledge_search
                - 用户搜索互联网上的最新信息、新闻、实时数据。当需要查询外部信息时使用 web_search，搜索一次即可，不要重复搜索
                - 问候类问题可以不使用工具直接回答
                请根据用户问题，选择合适的工具。

                - 调用 knowledge_search 后，**必须**基于工具返回的内容生成最终回答
                - 最终回答直接使用工具返回的内容，不要自己总结
                - 流程必须完整：用户提问 → 调用工具 → 获取结果 → 生成最终回答
                - **每次**用户询问公司制度相关问题（包括已经问过的），**都必须**重新调用 knowledge_search 获取最新信息，不要依赖历史对话
                - 不要因为之前回答过就跳过工具调用
                
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
