# 新增：RAG 业务编排——把流程串起来
import os
import shutil
from app.core.rag_engine import (
    load_document,
    split_texts,
    get_vector_store,
    create_qa_chain,
    delete_file_from_store,
    get_files_list,
    ask_question_stream,
)
from typing import AsyncIterator

UPLOAD_DIR = "./uploads"


# 上传文件
def upload(file):
    # 确保 ./uploads/ 目录存在，没有就新建，有的话就跳过
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    # 把用户上传的文件存到本地磁盘
    # 打开一个文件，准备写入，用二进制模式写——wb，w：写入文本，ab追加二进制
    with open(file_path, "wb") as buffer:
        # 把文件存到uploads下，是同步阻塞的 file.file->buffer
        shutil.copyfileobj(file.file, buffer)

    documents = load_document(file_path)
    chunks = split_texts(documents)
    vectordb = get_vector_store(chunks)
    return {
        "filename": file.filename,
        # 本次切了多少块
        "chunks": len(chunks),
        # 向量库有多少块
        "vector_count": vectordb._collection.count(),
        "message": "索引创建成功",
    }


def ask_question(question: str):
    # """基于向量库回答问题"""
    vectordb = get_vector_store()
    qa_chain = create_qa_chain(vectordb)
    result = qa_chain({"query": question})
    return {
        "answer": result.get("result", ""),
        "sources": list(
            set(
                [
                    doc.metadata.get("source", "未知来源")
                    for doc in result.get("source_documents", [])
                ]
            )
        ),
    }


# 删除文件
def delete_file(fileName: str) -> int:
    return delete_file_from_store(fileName)


# 获取文件列表
def get_files():
    files = get_files_list()
    return {"files": files, "total": len(files)}


# rag流式问答
async def ask_question_rag(question: str) -> AsyncIterator[str]:
    # ⬇️ 这里未来可以加逻辑
    # 1. 规则过滤（如敏感词）
    # 2. LLM 判断（如意图识别）
    # 3. 降级逻辑（如模型不可用用备用方案）
    # 4. 埋点/日志
    async for chunk in ask_question_stream(question):
        yield chunk
