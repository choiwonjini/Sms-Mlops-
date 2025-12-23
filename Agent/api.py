from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from agent_engine import TextOrderAgent
from schemas import AgentResponse

app = FastAPI(title="Agent 01 - Order Processor")

# Initialize Agent (Global State for Demo)
# In production, use session management.
agent = TextOrderAgent()

class ChatRequest(BaseModel):
    message: str # Simple message for query-based agent
    # Legacy fields optional
    messages: Optional[List[Dict[str, str]]] = None 
    guide_name: str = "order_guide"

@app.get("/")
def health_check():
    return {"status": "ok", "service": "Agent 01"}

@app.post("/chat")
def chat_endpoint(request: ChatRequest):
    try:
        # Extract message: either explicit 'message' field or last from history
        user_msg = request.message
        if not user_msg and request.messages:
            user_msg = request.messages[-1].get("content", "")
            
        response = agent.query(message=user_msg)
        return {"response": str(response)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    from config import settings
    uvicorn.run(app, host=settings.API_HOST, port=settings.API_PORT)
