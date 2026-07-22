import json
from fastapi import APIRouter, UploadFile, File, HTTPException, Query
from app.models.schemas import QuestionRequest
from app.services.rag_service import (
    upload,
    delete_file,
    ask_question_rag,
    normal_chat_stream,
    need_retrieval,
)
from app.services.file_service import (
    get_files_list,
)
from fastapi.responses import StreamingResponse
from langchain_core.runnables import RunnableConfig
from app.core.agent import get_agent, _profile_memory
from app.memory.extractor import UserInfoExtractor
import asyncio
from app.memory.user_profile import has_potential_info

router = APIRouter(prefix="/rag", tags=["RAG"])

_extractor = UserInfoExtractor()


# 上传文档 ，同一 session_id 上传多个文件，向量会累加
@router.post("/upload")
async def upload_file(
    file: UploadFile = File(...),
):
    try:
        upload(file)
        return {"code": 0, "message": "文件上传成功，已建立索引，可以开始提问"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


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
    try:
        print(f"要删除的文件：  {file_name}")
        delete_file(file_name)
        return {
            "code": 0,
            "message": "删除成功",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/agent")
async def agent_ask_stream(request: QuestionRequest):
    """
    Agent 流式问答
    - 实时输出 Agent 的思考和工具调用过程
    - 最终输出完整回答
    """

    async def generate():
        async def async_extract_and_update(user_id: str, messages: list):
            """后台异步执行：提取用户信息并更新画像"""
            try:
                print("提取用户画像:: ")
                extracted = _extractor.extract(messages)
                print(f"🔍 LLM 提取结果: {extracted}")
                if extracted.get("has_new_info", False):
                    _profile_memory.merge(user_id, extracted)
                    print(f"✅ 用户画像已更新: {extracted}")
            except Exception as e:
                print(f"❌ 提取用户信息失败: {e}")

        try:
            # ✅ 从请求中获取 user_id（暂时用默认值，后续可以从登录态获取）
            user_id = "user_001"
            agent = get_agent(user_id)

            # 用 thread_id 区分会话，
            config = RunnableConfig(
                {"configurable": {"thread_id": f"{user_id}_session_002"}}
            )

            # ✅ 使用 astream 流式执行
            async for chunk in agent.astream(
                {"messages": [("user", request.question)]},
                config=config,
                stream_mode="values",  # 每次状态变化都输出（用户消息、工具调用、AI 回复等）
            ):
                # 提取当前状态中的所有消息-累加的
                messages = chunk.get("messages", [])
                if messages:
                    last_msg = messages[-1]

                    # 判断消息类型
                    # ✅ 只输出 AI 的回答（AIMessage），跳过用户输入和工具消息
                    if hasattr(last_msg, "type"):
                        if (
                            last_msg.type == "ai"
                            and hasattr(last_msg, "content")
                            and last_msg.content
                        ):
                            yield f"data: {json.dumps({'type': 'message', 'content': last_msg.content})}\n\n"

                    # 工具调用信息保留（调试用）
                    if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
                        for tc in last_msg.tool_calls:
                            yield f"data: {json.dumps({'type': 'tool_call', 'tool': tc['name'], 'args': tc['args']})}\n\n"
            yield f"data: {json.dumps({'type': 'done'})}\n\n"

            # ✅ 生产级：提取用户信息（不阻塞流式输出）
            # 对话结束后，检查是否需要更新用户画像
            print("对话结束-done")
            final_state = agent.get_state(config)
            if final_state and final_state.values:
                messages = final_state.values.get("messages", [])
                user_messages = [
                    m for m in messages if hasattr(m, "type") and m.type == "human"
                ]

                if user_messages:
                    last_user_msg = user_messages[-1]

                    content = getattr(last_user_msg, "content", "")
                    print(f"对话结束-最后一条消息：{content}---")
                    if has_potential_info(content):
                        print("最近一条消息中有用户信息")
                        # ✅ 异步执行，不阻塞 done 的返回
                        asyncio.create_task(async_extract_and_update(user_id, messages))
            # 发送完成标记

        except Exception as e:
            import traceback

            traceback.print_exc()
            yield f"data: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")
