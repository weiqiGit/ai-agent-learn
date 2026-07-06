from fastapi import FastAPI
#pydantic是数据验证和设置管理库
from pydantic import BaseModel
import os
import requests

# 创建 FastAPI 实例
app = FastAPI()

# 定义请求体的格式
class ChatRequest(BaseModel):
    question: str

# AI 调用函数（从 hello_ai.py 迁移过来）
#定义一个函数，参数是question是str类型，返回值是str类型
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
        #注意question是从接口入参拿到的
        "messages": [{"role": "user", "content": question}],
    }
    response = requests.post(API_URL, headers=headers, json=payload)
    response.raise_for_status()
    data = response.json()
    return data["choices"][0]["message"]["content"]

# 路由：POST /chat，@是装饰器，作用是“把函数注册给路由”
@app.post("/chat")
def chat(request: ChatRequest):
    try:
        reply = call_deepseek(request.question)
        return {"reply": reply}
    except Exception as e:
        return {"error": str(e)}

# 路由：GET /，用来测试服务是否正常运行，可删
@app.get("/")
def read_root():
    return {"message": "AI 服务已启动，请访问 /docs 查看接口文档"}