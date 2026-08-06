import json
import os

from langchain_core.tools import tool

from app.core.rag_engine import get_vector_store
from app.tools.schemas import KnowledgeSearchInput


@tool(args_schema=KnowledgeSearchInput)
def knowledge_search(query: str) -> str:
    """
    查询公司内部文档、制度、流程、产品手册等。
    工具返回标准JSON字符串，结构：{"content": "检索文本内容", "sources": ["文档名1", "文档名2"]}
    若无数据则 content = "知识库中未找到相关信息"，sources = []
    """
    print(f"🔍 knowledge_search 开始执行，查询词: {query}")
    try:
        vectordb = get_vector_store()
        retriever = vectordb.as_retriever(search_kwargs={"k": 3})
        docs = retriever.invoke(query)

        if not docs:
            # 无结果，返回标准JSON
            return json.dumps(
                {"content": "知识库中未找到相关信息", "sources": []}, ensure_ascii=False
            )

        sources = list(
            {os.path.basename(doc.metadata.get("source", "未知来源")) for doc in docs}
        )
        content = "\n\n".join([doc.page_content for doc in docs])

        # 组装JSON并序列化返回
        resp_data = {"content": content, "sources": sources}
        result_json = json.dumps(resp_data, ensure_ascii=False)

        print(f"🔍 knowledge_search 返回JSON长度: {len(result_json)}")
        return result_json

    except Exception as e:
        error_msg = f"知识库检索失败：{str(e)}"
        print(f"❌ knowledge_search 异常: {error_msg}")
        # 异常同样遵循统一JSON结构
        return json.dumps({"content": error_msg, "sources": []}, ensure_ascii=False)
