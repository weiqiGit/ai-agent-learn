from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    question: str


class QuestionRequest(BaseModel):
    question: str


class SessionRequest(BaseModel):
    session_id: str


class UploadRequest(BaseModel):
    session_id: str


class FileInfo(BaseModel):
    name: str
    chunks: int
    source: str


class SQLPlaceholderInput(BaseModel):
    sql: str = Field(description="要执行的 SQL 查询语句，只能是 SELECT 查询")
