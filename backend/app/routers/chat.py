from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from app.services.rag_service import RagService
import json

router = APIRouter()

@router.websocket("/ws")
async def chat_ws(websocket: WebSocket):
    await websocket.accept()
    rag = RagService()
    try:
        while True:
            data = await websocket.receive_text()
            request = json.loads(data)
            async for token in rag.stream_answer(
                request["question"],
                request.get("history", []),
                request.get("document_id")
            ):
                await websocket.send_text(json.dumps({"token": token, "isFinal": False}))

            sources = rag.get_last_sources()
            await websocket.send_text(json.dumps({"token": "", "isFinal": True, "sources": sources}))
    except WebSocketDisconnect:
        pass