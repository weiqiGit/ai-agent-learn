from pydantic import BaseModel


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
