# app/tools/search.py
import httpx
from bs4 import BeautifulSoup
from urllib.parse import quote


def web_search(query: str) -> str:
    """使用百度搜索（国内直连）"""
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    }

    url = f"https://www.baidu.com/s?wd={quote(query)}"

    try:
        response = httpx.get(url, headers=headers, timeout=10.0, follow_redirects=True)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")
        results = []

        for item in soup.select(".result, .c-container"):
            title_elem = item.select_one("h3 a")
            if not title_elem:
                continue

            title = title_elem.get_text(strip=True)
            link = title_elem.get("href", "")

            snippet_elem = item.select_one(".c-abstract")
            snippet = snippet_elem.get_text(strip=True) if snippet_elem else "无摘要"

            results.append({"title": title, "snippet": snippet, "link": link})

            if len(results) >= 5:
                break

        if not results:
            return f"未找到 '{query}' 的相关结果"

        output = f"🔍 百度搜索 '{query}' 的结果：\n\n"
        for i, item in enumerate(results[:5], 1):
            output += (
                f"{i}. {item['title']}\n   {item['snippet']}\n   🔗 {item['link']}\n\n"
            )

        return output

    except httpx.TimeoutException:
        return "⏰ 搜索超时，请稍后重试"
    except Exception as e:
        return f"搜索失败：{str(e)}"
