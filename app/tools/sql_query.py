import re
import sqlite3
from typing import Any

# 数据库路径，和你初始化脚本保持一致
DB_PATH = "./data.db"
ALLOWED_TABLES = {"orders"}
# 只允许只读查询
READ_ONLY_PREFIX = "select"


def sql_query(args: dict[str, Any]) -> dict[str, Any]:
    """
    执行SQL查询工具函数
    :param args: dict，期望key: sql
    :return: {"content": 格式化文本结果, "sources": list[str]}
    """
    sql_str: str = args.get("sql", "").strip()
    sources = list(ALLOWED_TABLES)

    # -------------------------- 安全校验 --------------------------
    if not sql_str.lower().startswith(READ_ONLY_PREFIX):
        return {
            "content": "执行失败：仅支持 SELECT 查询，禁止增删改操作",
            "sources": sources,
        }

    # 简单防注入/危险操作过滤
    forbidden_keywords = {
        "insert",
        "update",
        "delete",
        "drop",
        "alter",
        "create",
        "truncate",
    }
    sql_lower = sql_str.lower()

    for kw in forbidden_keywords:
        # 匹配独立单词，前后是空格/括号/分号/开头结尾
        if re.search(rf"(^|\s|\(){kw}(\s|\)|;|$)", sql_lower):
            return {
                "content": f"执行失败：SQL 包含禁止关键字 [{kw}]，不允许执行",
                "sources": [],
            }

    conn = None
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row  # 支持按列名读取
        cursor = conn.cursor()
        cursor.execute(sql_str)
        rows = cursor.fetchall()

        if not rows:
            return {"content": "查询完成，未找到匹配数据", "sources": sources}

        # 组装表头
        columns = [desc[0] for desc in cursor.description]
        output_lines = ["|".join(columns)]
        output_lines.append("-" * 60)

        # 拼接数据（最多返回前50行，防止超长上下文炸LLM）
        limit_rows = rows[:50]
        for row in limit_rows:
            line_parts = [str(row[col]) for col in columns]
            output_lines.append("|".join(line_parts))

        if len(rows) > 50:
            output_lines.append(f"\n⚠️ 结果过多，仅展示前50条，总条数：{len(rows)}")

        content = "\n".join(output_lines)
        return {"content": content, "sources": sources}

    except sqlite3.Error as e:
        return {"content": f"SQL执行异常：{str(e)}", "sources": sources}
    finally:
        if conn:
            conn.close()
