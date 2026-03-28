from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from rag_service import handle_query

app = FastAPI()


@app.get("/")
def root():
    return {"status": "netsuite-ai running"}


@app.get("/v1/models")
def list_models():
    return {
        "object": "list",
        "data": [
            {
                "id": "netsuite-ai",
                "object": "model",
                "created": 1743134400,
                "owned_by": "openai"
            }
        ]
    }


@app.post("/v1/chat/completions")
async def chat(req: Request):
    try:
        body = await req.json()

        messages = body.get("messages", [])
        if not messages:
            return JSONResponse(
                status_code=400,
                content={"error": "No messages provided"}
            )

        user_msg = messages[-1].get("content", "")
        if not user_msg:
            return JSONResponse(
                status_code=400,
                content={"error": "Empty message"}
            )

        answer = handle_query(user_msg)

        return {
            "id": "netsuite-ai-response",
            "object": "chat.completion",
            "created": 1743134400,
            "model": "netsuite-ai",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": answer
                    },
                    "finish_reason": "stop"
                }
            ]
        }

    except Exception as e:
        print("APP ERROR:", str(e))
        return JSONResponse(
            status_code=500,
            content={
                "error": "Internal server error",
                "details": str(e)
            }
        )
