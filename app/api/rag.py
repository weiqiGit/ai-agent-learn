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
from app.core.agent import get_agent

router = APIRouter(prefix="/rag", tags=["RAG"])


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
async def agent_ask(request: QuestionRequest):
    try:
        agent = get_agent()

        async def generate():
            steps = []  # 存完整记录

            async for chunk in agent.astream({"input": request.question}):
                # 1. 实时推送工具调用
                if "steps" in chunk:
                    for step in chunk["steps"]:
                        action = step.action
                        observation = step.observation

                        tool_name = action.tool
                        tool_input = action.tool_input
                        steps.append(
                            {
                                "tool": tool_name,
                                "tool_input": tool_input,
                                "result": str(observation),
                            }
                        )

                        # 👇 实时推给前端
                        yield f"data: {json.dumps({'type': 'step', 'tool': tool_name, 'result': observation})}\n\n"

                # 2. 流式推送答案
                if "output" in chunk and chunk["output"]:
                    yield f"data: {json.dumps({'type': 'answer', 'content': chunk['output']})}\n\n"

            # 3. 最后发送完成信号（带完整步骤列表）
            yield f"data: {json.dumps({'type': 'done', 'steps': steps})}\n\n"

        return StreamingResponse(generate(), media_type="text/event-stream")

    except Exception as e:
        import traceback

        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
