from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

class ChatMessage(BaseModel):
    role: str = Field(..., description="system|user|assistant 중 하나")
    content: str

class ChatRequest(BaseModel):
    model: str = Field(..., description="gpt-4o, gpt-4.1, gpt-5, claude-3-5-sonnet, gemini-2.5-flash 등")
    messages: List[ChatMessage]
    temperature: float = 0.7
    max_tokens: Optional[int] = None

class ChatResponse(BaseModel):
    model: str
    output_text: str
    usage: Optional[Dict[str, Any]] = None
    raw: Optional[Dict[str, Any]] = None  # 디버깅용(프론트에선 숨겨도 됨)
    

