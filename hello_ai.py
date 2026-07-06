# os是python3的标准库，用于操作系统相关的操作
import os
# requests是python3的第三方库，用于发送http请求
import requests
# DeepSeek 提供的标准聊天接口
API_URL = "https://api.deepseek.com/v1/chat/completions"
# 从环境变量中获取 DeepSeek 的 API 密钥，设置： export DEEPSEEK_API_KEY="<KEY>"
API_KEY = os.getenv("DEEPSEEK_API_KEY")

if not API_KEY:
    raise RuntimeError("请设置环境变量 DEEPSEEK_API_KEY")
# 从控制台输入用户的问题，input()是python3的标准库，用于从控制台输入字符串，回车之后，赋值给user_input
user_input = input("请输入你的问题: ")

headers = {
    #f-string是python3的字符串格式化方法，用于将变量插入到字符串中，要不然就要做字符串拼接——"a"+b
    #bearer token用于API认证，Bearer是认证方式，token是认证凭据，Bearer token的格式为：Bearer <token>
    "Authorization": f"Bearer {API_KEY}",
    #不写的话，默认也是application/json
    "Content-Type": "application/json",
}

payload = {
    #模型名称
    "model": "deepseek-chat",
    "messages": [
        {"role": "user", "content": user_input},
    ],
}

# 发送请求，并获取响应，同步的，阻塞的
response = requests.post(API_URL, headers=headers, json=payload)
# 如果响应状态码不是200，则抛出异常
#response.raise_for_status()
#把 raise_for_status() 放在 try/except 里，可以打印出更详细的错误内容：
try:
    response.raise_for_status()
except requests.exceptions.HTTPError as e:
    print(f"请求失败: {e}")
    print(f"状态码: {response.status_code}")
    print(f"服务器返回内容: {response.text}")  # 这行很有用！
    #会看到红色异常栈，如果是调试环境可以直接看，生产环境可以去掉raise，直接退出屏蔽掉异常栈更干净
    #异常栈是 Python 在程序报错时打印的调用链，从报错位置追溯到入口函数，能帮我们快速定位是哪个文件、哪一行、因为什么原因出错的。
    raise SystemExit(1)
data = response.json()

#如果是调试，可以逐层打印
print(data.keys())  
print(f"\n {type(data['choices'])}")
ai_reply = data["choices"][0]["message"]["content"]
print(f"\nAI 回复: {ai_reply}")
