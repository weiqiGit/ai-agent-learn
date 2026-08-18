import json
import time

from fastapi import APIRouter, BackgroundTasks, File, HTTPException, Query, UploadFile
from fastapi.responses import StreamingResponse
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.errors import GraphInterrupt
from langgraph.graph import MessagesState
from langgraph.types import Command
from app.tools.schemas import ApprovalRequest


from app.core.agent import agent_graph
from app.core.exceptions import AppException
from app.memory.vector_memory import VectorMemory
from app.models.schemas import QuestionRequest
from app.services.file_service import (
    get_files_list,
)
from app.services.rag_service import (
    # ask_question_rag,
    delete_file,
    # need_retrieval,
    # normal_chat_stream,
    upload,
)
from app.utils.logger import logger

router = APIRouter(prefix="/rag", tags=["RAG"])


_vector_memory = VectorMemory()


# 上传文档 ，同一 session_id 上传多个文件，向量会累加
@router.post("/upload")
async def upload_file(
    file: UploadFile = File(...),
):
    if not file.filename:
        raise AppException(code=400, message="文件名不能为空", status_code=400)
    allowed_extensions = [".pdf", ".txt"]
    if not any(file.filename.endswith(ext) for ext in allowed_extensions):
        raise AppException(
            code=400,
            message=f"不支持的文件格式，支持: {', '.join(allowed_extensions)}",
            status_code=400,
        )
    upload(file)
    return {"code": 0, "message": "文件上传成功，已建立索引，可以开始提问"}


# 弃用-无tools问答
# @router.post("/ask")
# async def ask(
#     request: QuestionRequest,
# ):
#     try:
#         # need_retrieval通过关键词列表判断的
#         if need_retrieval(request.question):
#             return StreamingResponse(
#                 ask_question_rag(request.question), media_type="text/event-stream"
#             )
#         else:
#             return StreamingResponse(
#                 normal_chat_stream(request.question), media_type="text/event-stream"
#             )

#     except Exception as e:
#         return StreamingResponse(
#             # 转化成迭代器
#             iter([f"data: {json.dumps({'error': str(e)})}\n\n"]),
#             media_type="text/event-stream",
#         )


# 获取文件列表
@router.get("/files")
def get_list():
    try:
        fileInfo = get_files_list()
        return {"code": 0, "message": "success", "data": fileInfo}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# 删除文件
@router.delete("/deleteFile")
def delete_file_api(file_name: str = Query(..., description="要删除的文件名")):
    if not file_name or not file_name.strip():
        raise AppException(code=400, message="文件名不能为空", status_code=400)

    print(f"要删除的文件：  {file_name}")
    delete_file(file_name)
    return {
        "code": 0,
        "message": "删除成功",
    }


@router.post("/agent")
async def agent_ask_stream(
    request: QuestionRequest,
    background_tasks: BackgroundTasks,  # 👈 注入
):
    """
    Agent 流式问答
    - 实时输出 Agent 的思考和工具调用过程
    - 最终输出完整回答
    """

    async def generate():
        #  从请求中获取 user_id（暂时用默认值，后续可以从登录态获取）
        user_id = "user_001"
        start_time = time.time()
        print(f"⏰ [{time.strftime('%H:%M:%S')}] 请求开始")
        tool_calls = []
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
                print(
                    f"⏰ [{time.strftime('%H:%M:%S')}] Agent 已获取，耗时 {time.time() - start_time:.2f}s"
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
                    input_payload = MessagesState(
                        messages=[
                            HumanMessage(
                                content=f"请使用 sql_placeholder 工具查询数据库：{user_question}"
                            )
                        ]
                    )
                else:
                    input_payload = MessagesState(
                        messages=[HumanMessage(content=user_question)]
                    )
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
                        msg = event["message"]
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
    await agent_graph.ainvoke(
        Command(resume={"status": "rejected"}),
        config,
    )
    return {"ok": True}
