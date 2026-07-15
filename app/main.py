from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api import chat, rag
from dotenv import load_dotenv
import os
from fastapi.staticfiles import StaticFiles

load_dotenv()
app = FastAPI(title="AI Agent 服务")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


UPLOAD_DIR = "./uploads"
# 如果目录存在跳过，如果不存在创建目录
os.makedirs(UPLOAD_DIR, exist_ok=True)
# 挂载静态文件目录
app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")

# 注册路由，如果有多个，就注册多次，tags为了在docs里好看
app.include_router(chat.router, prefix="", tags=["chat"])
app.include_router(rag.router)
