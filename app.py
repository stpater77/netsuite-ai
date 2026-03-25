from fastapi import FastAPI
from pydantic import BaseModel
from chat import handle_user_message

app = FastAPI()

# =========================
# ORIGINAL CHAT ENDPOINT
# =========================
class ChatRequest(BaseModel):
    message: str

@app.post("/chat")
async def chat(req: ChatRequest):
    response = handle_user_message(req.message)
    return {"response": response}


# =========================
# OPENAI / OPENWEBUI COMPATIBLE ENDPOINT
# =========================
class OpenAIRequest(BaseModel):
    messages: list

@app.post("/v1/chat/completions")
async def openai_compatible(req: OpenAIRequest):
    # Get last user message
    user_message = req.messages[-1]["content"]

    # Process using your existing logic
    response_text = handle_user_message(user_message)

    # Return OpenAI-style response
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
# ROOT (OPTIONAL - FOR BROWSER TEST)
# =========================
@app.get("/")
def root():
    return {"message": "NetSuite AI is running 🚀"}
