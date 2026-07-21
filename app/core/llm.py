import httpx
import os
import requests
from typing import AsyncIterator


def call_deepseek(question: str) -> str:
    API_URL = "https://api.deepseek.com/v1/chat/completions"
    API_KEY = os.getenv("DEEPSEEK_API_KEY")

    if not API_KEY:
        raise ValueError("请设置环境变量 DEEPSEEK_API_KEY")

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": "deepseek-chat",
        # 注意question是从接口入参拿到的
        "messages": [{"role": "user", "content": question}],
    }
    response = requests.post(API_URL, headers=headers, json=payload)
    response.raise_for_status()
    # response返回的是json串格式，需要转换为dict类型
    data = response.json()
    return data["choices"][0]["message"]["content"]


# 手搓流式
# 返回值是一个迭代器，可以用for循环遍历，迭代器每个元素都是字符串类型
async def stream_deepseek(question: str) -> AsyncIterator[str]:
    API = "https://api.deepseek.com/v1/chat/completions"
    API_KEY = os.getenv("DEEPSEEK_API_KEY")
    if not API_KEY:
        raise ValueError("请设置环境变量 DEEPSEEK_API_KEY")
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": "deepseek-chat",
        "messages": [{"role": "user", "content": question}],
        "stream": True,
    }
    # 创建一个异步 HTTP 客户端，并在代码块结束后自动清理资源，要异步初始化连接池或进行 DNS 解析，这些操作都不应该阻塞事件循环
    async with httpx.AsyncClient(timeout=30.0) as client:
        # 等待服务器返回 HTTP 响应头（表明连接已建立），stream() 是一个异步上下文管理器，它会等待服务器返回 101 或 200 状态码，这个过程是 I/O 等待。
        async with client.stream(
            "POST", API, headers=headers, json=payload
        ) as response:
            response.raise_for_status()
            # 等待下一个数据块（\n）到达网络缓冲区
            async for line in response.aiter_lines():
                if line:
                    # 一旦 line 被 yield，如果前端消费慢，async for 会等待，协程会被挂起，不阻塞事件循环
                    # tcp“背压”（Backpressure）机制 —— 前端消费慢时，yield 被阻塞。
                    yield f"{line}\n\n"
