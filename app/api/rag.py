import json
import time

from fastapi import APIRouter, BackgroundTasks, File, HTTPException, Query, UploadFile
from fastapi.responses import StreamingResponse
from langchain_core.messages import AIMessageChunk, ToolMessage
from langchain_core.runnables import RunnableConfig

from app.core.agent import get_agent
from app.core.exceptions import AppException
from app.memory.user_profile import has_potential_info
from app.memory.vector_memory import VectorMemory
from app.models.schemas import QuestionRequest
from app.services.file_service import (
    get_files_list,
)
from app.services.rag_service import (
    ask_question_rag,
    async_extract_and_update,
    delete_file,
    need_retrieval,
    normal_chat_stream,
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


# 问答
@router.post("/ask")
async def ask(
    request: QuestionRequest,
):
    try:
        # need_retrieval通过关键词列表判断的
        if need_retrieval(request.question):
            return StreamingResponse(
                ask_question_rag(request.question), media_type="text/event-stream"
            )
        else:
            return StreamingResponse(
                normal_chat_stream(request.question), media_type="text/event-stream"
            )

    except Exception as e:
        return StreamingResponse(
            # 转化成迭代器
            iter([f"data: {json.dumps({'error': str(e)})}\n\n"]),
            media_type="text/event-stream",
        )


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
        try:
            #  用当前问题检索向量记忆
            memory_context = _vector_memory.get_context_prompt(
                user_id, request.question
            )
            agent = get_agent(user_id, memory_context)
            print(
                f"⏰ [{time.strftime('%H:%M:%S')}] Agent 已获取，耗时 {time.time() - start_time:.2f}s"
            )
            # 用 thread_id 区分会话，
            config = RunnableConfig(
                {
                    "configurable": {"thread_id": f"{user_id}_session_001"},
                    # 控制最大步数
                    "recursion_limit": 6,
                }
            )
            sources = []
            #  使用 astream 流式执行
            async for event in agent.astream_events(
                {"messages": [("user", request.question)]},
                config=config,
                version="v1",
            ):
                kind = event["event"]
                print(f"事件: {kind}")

                # LLM 生成 token
                if kind == "on_chat_model_stream":
                    chunk = event["data"].get("chunk")
                    # 必须取 .content 字符串，不要序列化整个 AIMessageChunk 对象
                    if chunk and chunk.content:
                        text = chunk.content
                        yield f"data: {json.dumps({'type': 'content', 'content': text}, ensure_ascii=False)}\n\n"

                # 工具开始调用（参数是完整的）
                elif kind == "on_tool_start":
                    tool_name = event["name"]
                    tool_input = event["data"].get("input", {})
                    print(f"🔧 调用工具: {tool_name}, 参数: {tool_input}")
                    tool_calls.append(tool_name)
                    yield f"data: {json.dumps({'type': 'tool_call', 'tool': tool_name, 'args': tool_input})}\n\n"

                # 工具调用结束
                elif kind == "on_tool_end":
                    tool_msg = event["data"].get("output")
                    if tool_msg and hasattr(tool_msg, "content"):
                        tool_out = tool_msg.content
                        try:
                            data = json.loads(tool_out)
                            sources = data.get("sources", [])
                            # 推送来源给前端
                            yield f"data: {json.dumps({'type': 'sources', 'sources': sources})}\n\n"
                        except json.JSONDecodeError:
                            # 兜底兼容旧格式
                            pass
            yield f"data: {json.dumps({'done': True})}\n\n"
            #  记录请求完成
            elapsed = time.time() - start_time
            logger.log(
                "request",
                {
                    "user_id": user_id,
                    "question": request.question,
                    "tool_calls": tool_calls,
                    "latency": round(elapsed, 3),
                    "status": "completed",
                    "desc": f"/agent接口请求结束：用户 {user_id} 提问完成，耗时 {round(elapsed, 3)}s",
                },
            )
            #  生产级：提取用户信息（不阻塞流式输出）
            # 对话结束后，检查是否需要更新用户画像
            print("对话结束-done")
            final_state = agent.get_state(config)
            if final_state and final_state.values:
                print(f"🔍 final_state.values 的 keys: {final_state.values.keys()}")
                print(f"🔍 final_state.values 完整内容: {final_state.values}")

                messages = final_state.values.get("messages", [])
                if messages:
                    # 只取用户消息
                    user_messages = [
                        m for m in messages if hasattr(m, "type") and m.type == "human"
                    ]
                    conversation_text = "\n".join(
                        [
                            f"{'用户' if hasattr(m, 'type') and m.type == 'human' else '助手'}：{getattr(m, 'content', '')}"
                            for m in messages
                        ]
                    )
                    if user_messages:
                        # 获取最近1条用户消息
                        recent_content = "\n".join(
                            [getattr(m, "content", "") for m in user_messages[-1:]]
                        )

                        # ✅ 每轮都检查当前消息是否包含关键词
                        print(f"🔍 检查消息: {recent_content[:50]}...")
                        print(f"🔍 message长度: {len(messages)}")
                        if has_potential_info(recent_content):
                            print("🔍 规则命中")

                            background_tasks.add_task(
                                async_extract_and_update,
                                user_id,
                                messages.copy(),
                            )
                        else:
                            print("⏭️ 规则未命中，跳过提取标签")

                    background_tasks.add_task(
                        _vector_memory.store,
                        user_id,
                        conversation_text,
                    )

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
