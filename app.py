from fastapi import FastAPI
from pydantic import BaseModel
from chat import handle_user_message

app = FastAPI()

# =========================
# BASIC CHAT ENDPOINT
# =========================
class ChatRequest(BaseModel):
    message: str

@app.post("/chat")
async def chat(req: ChatRequest):
    response = handle_user_message(req.message)
    return {"response": response}


# =========================
# OPENAI-COMPATIBLE CHAT ENDPOINT
# =========================
class OpenAIRequest(BaseModel):
    messages: list

@app.post("/v1/chat/completions")
async def openai_compatible(req: OpenAIRequest):
    user_message = req.messages[-1]["content"]

    response_text = handle_user_message(user_message)

    return {
        "id": "chatcmpl-123",
        "object": "chat.completion",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": response_text
                },
                "finish_reason": "stop"
            }
        ]
    }


# =========================
# REQUIRED FOR OPENWEBUI (THIS FIXES YOUR ISSUE)
# =========================
@app.get("/v1/models")
def list_models():
    return {
        "object": "list",
        "data": [
            {
                "id": "netsuite-ai",
                "object": "model",
                "owned_by": "netsuite-ai"
            }
        ]
    }


# =========================
# ROOT ENDPOINT (OPTIONAL)
# =========================
@app.get("/")
def root():
    return {"message": "NetSuite AI is running 🚀"}
