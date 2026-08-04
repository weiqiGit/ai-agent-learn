import httpx
from urllib.parse import quote

url = f'https://www.baidu.com/s?wd={quote("Python")}'
headers = {
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
}

response = httpx.get(url, headers=headers, timeout=10.0, follow_redirects=True)
print('状态码:', response.status_code)
print('内容前800字符:')
print(response.text[:800])
