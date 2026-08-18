import json
import os
from datetime import datetime, timezone
from typing import Literal, Optional, cast

from dotenv import load_dotenv
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import StructuredTool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.config import get_stream_writer
from langgraph.graph import START, StateGraph
from langgraph.graph.message import MessagesState
from langgraph.types import interrupt
from pydantic import SecretStr

# 给LLM暴露的全部工具：增加占位工具，真实sql_query不在这里！
# 业务模块导入
from app.memory.user_profile import UserProfileMemory
from app.models.schemas import SQLPlaceholderInput
from app.tools import calculator, knowledge_search, web_search
from app.tools.sql_query import sql_query

load_dotenv()

# ===================== 全局配置 =====================
BASE_SYSTEM_PROMPT = """你是一个智能助手。当用户提出问题时，必须考虑是否需要使用工具。
1. 用户没有明确询问公司信息时，不要主动提及公司
2. 只回答用户直接问的问题，不要过度延伸
3. 记住用户说的个人信息，但不要自己发挥

工具规则：
1. knowledge_search：查询公司内部文档、制度、流程、产品手册。询问公司相关内容必须调用
2. web_search：搜索互联网最新信息、新闻、实时数据，外部信息使用，不要重复搜索
3. calculator：数学计算，涉及计算必须调用工具，禁止心算
4. sql_placeholder：提交数据库查询申请，用于数据分析统计。该申请需要人工审批，不能直接获取数据。

问候类闲聊可直接回答不调用工具。
"""

_profile_memory = UserProfileMemory()
_checkpointer = InMemorySaver()

# 普通工具（可直接执行）
normal_tools = [
    knowledge_search,
    web_search,
    calculator,
]


all_available_tools = [
    *normal_tools,
    StructuredTool.from_function(
        lambda sql: None,  # 永远不会执行，仅占位
        name="sql_placeholder",
        description="""
提交数据库查询申请，用于数据分析、统计、报表。提交后需要人工审批，审批通过才会执行查询。

【数据库只读表清单，仅允许使用下列表，禁止编造表名、字段】
# orders 订单表
order_id TEXT：订单编号（主键）
user_id TEXT：用户编号
city TEXT：城市名称，可选：北京、上海、广州、深圳、杭州、成都、武汉
category TEXT：商品品类：数码、服饰、生鲜、家电、图书、美妆、食品
order_amount REAL：订单金额
order_status TEXT：订单状态：paid(已支付)、refunded(已退款)、cancelled(已取消)
create_date TEXT：下单日期，格式 YYYY-MM-DD
pay_channel TEXT：支付渠道：微信、支付宝、银行卡

SQL编写强制规则：
1. 只允许 SELECT 查询，不能写 INSERT / UPDATE / DELETE / DROP / ALTER
2. 返回的SQL 语句要格式化（换行、缩进）
3. 禁止 SELECT *，明确写出需要查询的字段
4. 日期条件直接使用字符串匹配，不要使用复杂日期函数
5. 不要一次性查询超大数量数据，必要时合理聚合统计
6. 【当前日期】会通过系统提示注入，"上个月"、"今年"等相对时间请自行推算，不要反问用户。
7. 用户没有指定时间范围时，可以主动询问用户，不要擅自扩大查询区间

【数据展示规范】
- 查询结果是多条明细时，用**编号列表**展示，不要用 Markdown 表格
- 格式示例：
  1. 订单号 ORD001：武汉，1263.32元，2026-01-23，支付宝
  2. 订单号 ORD002：北京，88.00元，2026-01-15，微信
- 每条数据一行，字段之间用中文逗号分隔
- 数据超过 20 条时只展示前 20 条并说明总数
""",
        args_schema=SQLPlaceholderInput,
    ),
]


# 初始化LLM
def build_llm():
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise ValueError("请设置环境变量 DEEPSEEK_API_KEY")
    return ChatOpenAI(
        model="deepseek-chat",
        api_key=SecretStr(api_key),
        base_url="https://api.deepseek.com/v1",
        temperature=0.3,
    )


llm = build_llm()
llm_with_tools = llm.bind_tools(all_available_tools)


# ===================== 节点定义 =====================
async def call_llm_node(
    state: MessagesState,
    config: RunnableConfig,
):
    writer = get_stream_writer()
    messages = state["messages"]
    configurable = config.get("configurable", {})
    user_id = configurable.get("user_id", "default")
    memory_context = configurable.get("memory_context", "")

    # ========== 你的Prompt逻辑保持原样 ==========
    user_context = _profile_memory.get_context_prompt(user_id)
    full_system_text = BASE_SYSTEM_PROMPT
    today = datetime.now(timezone.utc).strftime("%Y年%m月%d日")
    full_system_text += f"\n【当前日期】{today}"  # ← 加这行
    full_system_text += (
        f"\n【用户信息】\n{user_context if user_context else '暂无用户信息'}"
    )
    full_system_text += (
        f"\n【用户记忆】\n{memory_context if memory_context else '暂无用户记忆'}"
    )
    full_messages = [SystemMessage(content=full_system_text)] + messages

    full_chunk: Optional[AIMessageChunk] = None
    print("\n=== call_llm_node messages ===")
    for i, msg in enumerate(state["messages"]):
        t = type(msg).__name__
        tcs = getattr(msg, "tool_calls", None)
        tcid = getattr(msg, "tool_call_id", None)
        content = str(msg.content)[:60]
        print(
            f"[{i}] {t} | tool_calls={bool(tcs)} | tool_call_id={tcid} | content={content}"
        )
    print("==============================\n")

    async for raw_chunk in llm_with_tools.astream(full_messages):
        # 过滤非AI流块（防御）
        if not isinstance(raw_chunk, AIMessageChunk):
            continue
        chunk: AIMessageChunk = raw_chunk

        # 下发打字文本
        if chunk.content:
            writer({"type": "content", "content": chunk.content})

        # 合并逻辑修复点：分开写，cast合并结果
        if full_chunk is None:
            full_chunk = chunk
        else:
            combined: BaseMessage = full_chunk + chunk
            full_chunk = cast(AIMessageChunk, combined)

    if full_chunk is None:
        raise RuntimeError("LLM 未返回任何输出 chunk")

    ai_msg = AIMessage(
        content=full_chunk.content,
        tool_calls=full_chunk.tool_calls,
        id=full_chunk.id,
        usage_metadata=full_chunk.usage_metadata,
    )

    return {"messages": [ai_msg]}


def wrapped_tool_node(
    state: MessagesState,
):
    writer = get_stream_writer()
    tools_by_name = {tool.name: tool for tool in normal_tools}
    last_msg = state["messages"][-1]
    if not isinstance(last_msg, AIMessage) or not last_msg.tool_calls:
        return "__end__"
    tool_calls = last_msg.tool_calls
    responses = []

    for tool_call in tool_calls:
        tool = tools_by_name[tool_call["name"]]
        writer(
            {
                "event": "on_tool_start",
                "name": tool_call["name"],
                "input": tool_call["args"],
                "tool_call_id": tool_call["id"],
            }
        )

        result = tool.invoke(tool_call["args"])
        writer(
            {
                "event": "on_tool_end",
                "name": tool_call["name"],
                "tool_call_id": tool_call["id"],
                "output": result,
            }
        )
        responses.append(
            ToolMessage(
                content=str(result),
                tool_call_id=tool_call["id"],
                name=tool_call["name"],
            )
        )
    return {"messages": responses}


async def sql_approval_node(state: MessagesState):
    writer = get_stream_writer()

    # ========== 1. 找带 tool_calls 的 AIMessage ==========
    last_msg = None
    for msg in reversed(state["messages"]):
        if isinstance(msg, AIMessage) and msg.tool_calls:
            last_msg = msg
            break

    if not isinstance(last_msg, AIMessage):
        raise TypeError("sql_approval_node 仅能由模型工具调用消息进入")

    tool_call = last_msg.tool_calls[0]
    tool_call_id = tool_call["id"]
    query_args = tool_call["args"]
    tool_name = tool_call["name"]

    # ========== 2. 推送审批开始事件 ==========
    writer(
        {
            "event": "sql_apply_start",
            "name": tool_name,
            "input": query_args,
            "tool_call_id": tool_call_id,
        }
    )

    interrupt_data = {
        "type": "sql_approval",
        "tool": tool_name,
        "args": query_args,
        "tool_call_id": tool_call_id,
        "message": "即将提交数据库查询申请，请管理员确认是否允许执行",
    }
    writer({"event": "__interrupt__", "data": interrupt_data})

    # ========== 3. 暂停，等待用户审批 ==========
    approval_result = interrupt(interrupt_data)

    is_rejected = (
        not approval_result
        or isinstance(approval_result, dict)
        and approval_result.get("status") == "rejected"
    )

    # ========== 4. 审批拒绝：直接返回驳回 ToolMessage ==========
    if is_rejected:
        writer(
            {
                "event": "sql_approval_done",
                "name": tool_name,
                "tool_call_id": tool_call_id,
                "status": "rejected",
                "output": "本次数据库查询申请已被驳回",
            }
        )
        return {
            "messages": [
                ToolMessage(
                    content="本次数据库查询申请已被驳回",
                    tool_call_id=tool_call_id,
                    name="sql_placeholder",
                )
            ]
        }

    # ========== 5. 审批通过：直接执行 SQL（原 sql_execute_node 逻辑）==========
    writer(
        {
            "event": "sql_approval_done",
            "name": tool_name,
            "tool_call_id": tool_call_id,
            "status": "approved",
            "output": "开始执行SQL查询",
        }
    )

    # 直接用 query_args，不再从 ToolMessage 解析 [AllowExecute]
    result = sql_query(query_args)

    writer(
        {
            "event": "sql_query_done",
            "name": "sql_placeholder",
            "tool_call_id": tool_call_id,
            "output": result,
        }
    )

    # ========== 6. 只返回这一条 ToolMessage ==========
    return {
        "messages": [
            ToolMessage(
                content=str(result),
                tool_call_id=tool_call_id,
                name="sql_placeholder",
            )
        ]
    }


# ===================== 路由函数 =====================
def route_after_llm(
    state: MessagesState,
) -> Literal["tool_node", "sql_approval_node", "__end__"]:
    last_msg = state["messages"][-1]
    if not isinstance(last_msg, AIMessage) or not last_msg.tool_calls:
        return "__end__"

    tool_name = last_msg.tool_calls[0]["name"]
    if tool_name == "sql_placeholder":
        return "sql_approval_node"
    return "tool_node"


# ===================== 构建图 =====================
def build_agent_graph():
    builder = StateGraph(MessagesState)

    builder.add_node("call_llm_node", call_llm_node)
    builder.add_node("tool_node", wrapped_tool_node)
    builder.add_node("sql_approval_node", sql_approval_node)

    builder.add_edge(START, "call_llm_node")
    builder.add_conditional_edges("call_llm_node", route_after_llm)
    builder.add_edge("tool_node", "call_llm_node")
    builder.add_edge("sql_approval_node", "call_llm_node")

    graph = builder.compile(checkpointer=_checkpointer)
    return graph


agent_graph = build_agent_graph()
