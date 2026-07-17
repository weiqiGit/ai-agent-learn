from app.core.rag_engine import get_vector_store


# 搜索知识库
def knowledge_search(query: str) -> str:
    try:
        vectordb = get_vector_store()
        retriever = vectordb.as_retriever(search_kwargs={"k": 3})
        docs = retriever.invoke(query)

        if not docs:
            return "知识库中未找到相关信息"

        context = "\n\n".join([doc.page_content for doc in docs])
        return f"找到以下相关信息：\n{context}"
    except Exception as e:
        return f"知识库检索失败：{str(e)}"
