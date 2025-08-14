import os
from typing import Dict, Any, List
from anthropic import Anthropic

client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

# 모델 예: claude-3-5-sonnet-latest, claude-3-opus-latest 등

def _to_claude_messages(messages: List[dict]):
    converted = []
    system_texts = []
    for m in messages:
        role = m["role"]
        if role == "system":
            system_texts.append(m["content"])
        elif role in ("user", "assistant"):
            converted.append({"role": role, "content": m["content"]})
    system = "\n".join(system_texts) if system_texts else None
    return system, converted

def run_anthropic(model: str, messages, temperature: float, max_tokens: int | None) -> Dict[str, Any]:
    system, conv = _to_claude_messages(messages)

    resp = client.messages.create(
        model=model,
        system=system,
        messages=conv,
        temperature=temperature,
        max_tokens=max_tokens or 1024,
    )

    text = "".join([b.text for b in resp.content if getattr(b, "type", "") == "text"])  # 단순 텍스트만

    # usage 는 토큰 정보가 들어있음
    usage = {
        "input_tokens": getattr(resp.usage, "input_tokens", None),
        "output_tokens": getattr(resp.usage, "output_tokens", None),
    }

    return {"output_text": text, "usage": usage, "raw": resp.model_dump()}