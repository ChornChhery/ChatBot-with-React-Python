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

            # FIX: Safely parse request — bad JSON or missing "question" key
            # will no longer kill the entire WebSocket connection
            try:
                request = json.loads(data)
                question = request.get("question", "").strip()
                if not question:
                    await websocket.send_text(json.dumps({
                        "token": "Error: question is required.",
                        "isFinal": True,
                        "sources": [],
                        "ragUsed": False
                    }))
                    continue
            except json.JSONDecodeError:
                await websocket.send_text(json.dumps({
                    "token": "Error: invalid message format.",
                    "isFinal": True,
                    "sources": [],
                    "ragUsed": False
                }))
                continue

            # FIX: Catch errors during streaming (Ollama down, embed failure, etc.)
            # so the WebSocket stays alive and reports the error to the frontend
            try:
                async for token in rag.stream_answer(
                    question,
                    request.get("history"),       # None-safe, rag_service handles it
                    request.get("document_id")
                ):
                    await websocket.send_text(json.dumps({"token": token, "isFinal": False}))

                sources = rag.get_last_sources()
                rag_used = rag.get_rag_used()

                # FIX: Send rag_used flag so frontend knows whether to show sources
                await websocket.send_text(json.dumps({
                    "token": "",
                    "isFinal": True,
                    "sources": sources,
                    "ragUsed": rag_used      # True = show sources, False = LLM-only answer
                }))

            except Exception as e:
                print(f"[WebSocket] Stream error: {e}")
                await websocket.send_text(json.dumps({
                    "token": f"Error: {str(e)}",
                    "isFinal": True,
                    "sources": [],
                    "ragUsed": False
                }))

    except WebSocketDisconnect:
        pass