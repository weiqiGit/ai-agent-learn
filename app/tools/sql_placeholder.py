from langchain_core.tools import tool


@tool
def sql_placeholder(question: str):
    """
    需要查询业务数据库结构化数据时调用此工具
    Args:
        question: 用户想要查询数据的自然语言描述
    """
    raise NotImplementedError("sql_placeholder 仅作为标记工具，禁止执行")
