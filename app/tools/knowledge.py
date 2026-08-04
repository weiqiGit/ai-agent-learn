import os

from langchain_core.tools import tool

from app.core.rag_engine import get_vector_store
from app.tools.schemas import KnowledgeSearchInput


@tool(args_schema=KnowledgeSearchInput)
def knowledge_search(query: str) -> str:
    """查询公司内部文档、制度、流程、产品手册等。"""
    print(f"🔍 knowledge_search 开始执行，查询词: {query}")
    try:
        vectordb = get_vector_store()
        retriever = vectordb.as_retriever(search_kwargs={"k": 3})
        docs = retriever.invoke(query)

        if not docs:
            return "知识库中未找到相关信息"

        sources = list(
            {os.path.basename(doc.metadata.get("source", "未知来源")) for doc in docs}
        )
        context = "\n\n".join([doc.page_content for doc in docs])
        result = context + f"\n\n📎 来源：{', '.join(sources)}"
        print(f"🔍 knowledge_search 返回内容长度: {len(result)}")
        return result

    except Exception as e:  # noqa: BLE001
        error_msg = f"知识库检索失败：{e!s}"
        print(f"❌ knowledge_search 异常: {error_msg}")
        return error_msg
