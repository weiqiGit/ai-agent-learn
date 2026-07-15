from fastapi import APIRouter
from fastapi.responses import StreamingResponse
import json
from app.models.schemas import ChatRequest
from app.core.llm import stream_deepseek
from app.core.llm import call_deepseek


router = APIRouter()

@router.post("/chat")
def chat(request: ChatRequest):
    try:
        reply = call_deepseek(request.question)
        return {"reply": reply}
    except Exception as e:
        return {"error": str(e)}


@router.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    async def generate_response():
        try:
           async for chunk in stream_deepseek(request.question): 
               yield chunk
        except Exception as e:
            yield f"{json.dumps({'error': str(e)})}\n" 
    #StreamingResponse需要传入两个参数：一个是流对象，一个是媒体类型，做了序列化
    return StreamingResponse(generate_response(),media_type="text/event-stream")
    