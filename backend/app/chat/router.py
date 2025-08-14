from fastapi import APIRouter, HTTPException
from ..schemas import ChatRequest, ChatResponse
from .adapters.openai_adapter import run_openai
from .adapters.anthropic_adapter import run_anthropic
from .adapters.gemini_adapter import run_gemini

router = APIRouter(prefix="/chat", tags=["chat"])

# 간단 라우팅: 모델 이름에 따라 어댑터 선택

def _pick_vendor(model: str) -> str:
    m = model.lower()
    if m.startswith("gpt"):  # OpenAI
        return "openai"
    if m.startswith("claude"):  # Anthropic
        return "anthropic"
    if m.startswith("gemini"):  # Google
        return "gemini"
    # 기타는 나중에 추가
    return "unknown"

@router.post("/messages", response_model=ChatResponse)
async def chat_messages(req: ChatRequest):
    vendor = _pick_vendor(req.model)

    if vendor == "openai":
        data = run_openai(req.model, [m.model_dump() for m in req.messages], req.temperature, req.max_tokens)
    elif vendor == "anthropic":
        data = run_anthropic(req.model, [m.model_dump() for m in req.messages], req.temperature, req.max_tokens)
    elif vendor == "gemini":
        data = run_gemini(req.model, [m.model_dump() for m in req.messages], req.temperature, req.max_tokens)
    else:
        raise HTTPException(400, detail="지원하지 않는 모델 이름입니다.")

    return ChatResponse(model=req.model, **data)