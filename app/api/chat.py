import json

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.errors import GraphInterrupt
from langgraph.types import Command

from app.core.agent import agent_graph
from app.core.llm import call_deepseek, stream_deepseek
from app.memory.vector_memory import VectorMemory
from app.models.schemas import ChatRequest, QuestionRequest
from app.tools.schemas import ApprovalRequest
from typing import Annotated, Literal, NotRequired, Optional, TypedDict, cast
from langgraph.graph.message import add_messages

from app.utils.logger import logger

router = APIRouter()
_vector_memory = VectorMemory()


class MessagesState(TypedDict):
    messages: Annotated[list, add_messages]
    approval_pushed: NotRequired[bool]


# 主要在用
@router.post("/agent")
async def agent_ask_stream(
    request: QuestionRequest,
):
    """
    Agent 流式问答
    - 实时输出 Agent 的思考和工具调用过程
    - 最终输出完整回答
    """

    async def generate():
        #  从请求中获取 user_id（暂时用默认值，后续可以从登录态获取）
        user_id = "user_001"
        # 记录请求开始
        logger.log(
            "request",
            {
                "user_id": user_id,
                "question": request.question,
                "status": "started",
                "desc": f"/agent接口请求开始，用户 {user_id} 开始提问",
            },
        )
        force_query = False
        user_question = request.question
        print(f"request.question.lower():{request.question.lower()}")
        # ✅ 检测 query: 前缀
        if request.question.lower().startswith("query:"):
            user_question = request.question[6:].strip()
            print(f"请使用 sql_placeholder 工具查询数据库：{user_question}")
            force_query = True
        if not user_question:
            raise HTTPException(status_code=400, detail="查询内容不能为空")
        try:
            #  用当前问题检索向量记忆
            memory_context = _vector_memory.get_context_prompt(
                user_id, request.question
            )
            thread_id = f"{user_id}_session_001"
            # 用 thread_id 区分会话，
            config = RunnableConfig(
                {
                    "configurable": {
                        "thread_id": thread_id,
                        "memory_context": memory_context,
                    },
                    # 控制最大步数
                    "recursion_limit": 10,
                }
            )
            sources = []
            if force_query:
                input_payload: MessagesState = {
                    "messages": [
                        HumanMessage(
                            content=f"请使用 sql_placeholder 工具查询数据库：{user_question}"
                        )
                    ],
                    "approval_pushed": False,
                }
            else:
                input_payload: MessagesState = {
                    "messages": [HumanMessage(content=user_question)],
                    "approval_pushed": False,
                }
                # 第一个事件先把 thread_id 推给前端
            yield f"data: {json.dumps({'type': 'thread_id', 'thread_id': thread_id})}\n\n"
            #  使用 astream 流式执行
            async for mode, data in agent_graph.astream(
                input_payload,
                config=config,
                stream_mode=["custom"],
            ):
                if not isinstance(data, dict):
                    continue

                # ========== 类型1：LLM 流式 token（打字机效果）==========
                if data.get("type") == "content":
                    yield f"data: {json.dumps({'type': 'token', 'content': data['content']}, ensure_ascii=False)}\n\n"
                    continue

                # ========== 类型2：你 writer 推送的自定义事件 ==========
                if "event" not in data:
                    continue

                event = data
                kind = event["event"]
                print(f"事件: {kind}")

                if kind == "__interrupt__":
                    yield f"data: {json.dumps({'type': 'interrupt', 'data': event['data']}, ensure_ascii=False)}\n\n"
                    break

                elif kind == "sql_apply_start":
                    payload = {
                        "type": "tool_call",
                        "tool": event["name"],
                        "args": event["input"],
                    }
                    yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

                elif kind == "sql_approval_done":
                    status = event["status"]
                    msg = event["output"]
                    print(f"📌 SQL审批完成，状态:{status}, {msg}")
                    yield f"data: {json.dumps({'type': 'notify', 'text': msg}, ensure_ascii=False)}\n\n"

                elif kind == "sql_query_done":
                    tool_output = event["output"]
                    if isinstance(tool_output, dict) and "sources" in tool_output:
                        sources = tool_output["sources"]
                        yield f"data: {json.dumps({'type': 'sources', 'sources': sources}, ensure_ascii=False)}\n\n"

                elif kind == "on_tool_start":
                    payload = {
                        "type": "tool_call",
                        "tool": event["name"],
                        "args": event["input"],
                    }
                    yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

                elif kind == "on_tool_end":
                    tool_name = event["name"]
                    tool_output = event["output"]
                    print(f"✅ 工具结束: {tool_name}")

                    try:
                        if isinstance(tool_output, str):
                            parsed = json.loads(tool_output)
                            if "sources" in parsed:
                                sources = parsed["sources"]
                                yield f"data: {json.dumps({'type': 'sources', 'sources': sources}, ensure_ascii=False)}\n\n"
                    except json.JSONDecodeError:
                        pass
            yield f"data: {json.dumps({'done': True})}\n\n"
        except GraphInterrupt as e:
            # ✅ 中断信号
            yield f"data: {json.dumps({'type': 'interrupt', 'data': e.args[0]})}\n\n"

        except Exception as e:
            import traceback

            traceback.print_exc()
            #  记录错误
            logger.log(
                "error",
                {
                    "user_id": user_id,
                    "question": request.question,
                    "error": str(e),
                    # 完整错误堆栈，可能暴露代码路径和内部逻辑，通常记录在服务端日志里，不返回给前端
                    "traceback": traceback.format_exc(),
                    "desc": f"用户 {user_id} 请求失败",
                },
            )
            yield f"data: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


@router.post("/agent/sql/approve")
async def approve_sql(request: ApprovalRequest):
    """用户点击确认"""
    thread_id = request.thread_id
    config: RunnableConfig = {"configurable": {"thread_id": thread_id}}

    async def event_generator():
        async for chunk in agent_graph.astream(
            Command(resume={"status": "approved"}),
            config,
            stream_mode=["custom"],
        ):
            data = chunk[1] if isinstance(chunk, tuple) else chunk
            if not isinstance(data, dict):
                continue
            # LLM token 输出（call_llm_node 里 writer 推送的 content）
            if data.get("type") == "content":
                yield f"data: {json.dumps({'type': 'token', 'content': data['content']}, ensure_ascii=False)}\n\n"
                continue
            event = data["event"]

            if event == "on_tool_end":
                tool_output = data.get("output")
                try:
                    if isinstance(tool_output, str):
                        parsed = json.loads(tool_output)
                        if "sources" in parsed:
                            yield f"data: {json.dumps({'type': 'sources', 'sources': parsed['sources']}, ensure_ascii=False)}\n\n"
                except json.JSONDecodeError:
                    pass

            elif event == "sql_query_done":
                tool_output = data.get("output")
                if isinstance(tool_output, dict) and "sources" in tool_output:
                    yield f"data: {json.dumps({'type': 'sources', 'sources': tool_output['sources']}, ensure_ascii=False)}\n\n"

            if "event" not in data:
                continue

        yield f"data: {json.dumps({'done': True})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@router.post("/agent/sql/reject")
async def reject_sql(request: ApprovalRequest):
    """用户点击取消"""
    thread_id = request.thread_id
    config: RunnableConfig = {"configurable": {"thread_id": thread_id}}
    result = await agent_graph.ainvoke(
        Command(resume={"status": "rejected"}),
        config,
    )

    # 取最后一条消息（LLM 的回复）
    last_msg = result["messages"][-1]
    return {"ok": True, "answer": last_msg.content}


# 非流式-普通对话
@router.post("/chat")
def chat(request: ChatRequest):
    try:
        reply = call_deepseek(request.question)
        return {"reply": reply}
    except Exception as e:
        return {"error": str(e)}


# 流式-普通对话
@router.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    async def generate_response():
        try:
            async for chunk in stream_deepseek(request.question):
                yield chunk
        except Exception as e:
            yield f"{json.dumps({'error': str(e)})}\n"

    # StreamingResponse需要传入两个参数：一个是流对象，一个是媒体类型，做了序列化
    return StreamingResponse(generate_response(), media_type="text/event-stream")
