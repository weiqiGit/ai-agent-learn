import os
import time

import httpx
from langchain_core.tools import tool
from tavily import TavilyClient

from app.tools.schemas import WebSearchInput
from app.utils.retry import retry

# 初始化客户端
tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))


def on_retry(name, attempt, error, delay):
    print(f"⏳ {name} 重试 {attempt} 次，等待 {delay:.2f}s，错误: {error}")


@tool(args_schema=WebSearchInput)
@retry(
    max_retries=3,
    base_delay=1.0,
    backoff=2.0,
    max_delay=30.0,
    jitter=0.2,
    exceptions=(httpx.TimeoutException, httpx.ConnectError),
    on_retry=on_retry,
)
def web_search(query: str) -> str:
    """搜索互联网上的最新信息、新闻、实时数据。"""
    start = time.time()
    print(f"⏰ [{time.strftime('%H:%M:%S')}] web_search 开始: {query}")
    """使用百度搜索（国内直连）"""
    try:
        if not query or len(query.strip()) < 2:
            return "搜索词太短，请提供更具体的关键词"
        if not any(word in query for word in ["中文", "中国", "国内"]):
            query = f"{query} 中文"

        response = tavily.search(
            query=query,
            search_depth="basic",  # basic 或 advanced
            max_results=2,
            include_answer=True,
        )

        # 如果有 AI 总结的答案，优先展示
        if response.get("answer"):
            return f"🔍 搜索结果（AI 总结）：\n{response['answer']}"

        # 否则展示原始结果
        results = response.get("results", [])
        elapsed = time.time() - start
        print(f"⏰ [{time.strftime('%H:%M:%S')}] web_search 完成，耗时 {elapsed:.2f}s")
        if not results:
            return f"未找到 '{query}' 的相关结果"

        output = f"🔍 搜索 '{query}' 的结果：\n\n"
        for i, item in enumerate(results[:5], 1):
            output += f"{i}. {item.get('title', '无标题')}\n"
            output += f"   {item.get('content', '无内容')[:200]}...\n"
            output += f"   🔗 {item.get('url', '')}\n\n"

        return output

    except Exception as e:
        return f"搜索失败：{str(e)}"
