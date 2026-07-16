import json
from fastapi import APIRouter, UploadFile, File, HTTPException, Query
from app.models.schemas import QuestionRequest
from app.services.rag_service import (
    upload,
    delete_file,
    get_files,
    ask_question_rag,
    normal_chat_stream,
    need_retrieval,
)
from fastapi.responses import StreamingResponse


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
        fileInfo = get_files()
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
