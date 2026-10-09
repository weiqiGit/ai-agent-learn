from fastapi import APIRouter, File, HTTPException, Query, UploadFile

from app.core.exceptions import AppException
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

router = APIRouter()


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
