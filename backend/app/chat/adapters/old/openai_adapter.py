import os
from typing import Dict, Any
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# OpenAI는 responses API 사용(텍스트 기준)
# 모델 예: gpt-4o, gpt-4.1, gpt-5 (계정 보유 모델만 동작)

def run_openai(model: str, messages, temperature: float, max_tokens: int | None) -> Dict[str, Any]:
    # messages -> responses.create 의 input 으로 단순 변환
    # system 프롬프트가 있으면 instructions 로, 나머지는 교대로 합치기(간단 버전)
    system_text = "\n".join([m["content"] for m in messages if m["role"] == "system"]) or None
    user_contents = []
    for m in messages:
        if m["role"] in ("user", "assistant"):
            prefix = "User:" if m["role"] == "user" else "Assistant:"
            user_contents.append(f"{prefix} {m['content']}")
    input_text = "\n".join(user_contents)

    resp = client.responses.create(
        model=model,
        instructions=system_text,
        input=input_text,
        temperature=temperature,
        max_output_tokens=max_tokens,
    )

    # 통일된 형태로 리턴
    return {
        "output_text": resp.output_text,
        "usage": getattr(resp, "usage", None).model_dump() if hasattr(resp, "usage") else None,
        "raw": resp.model_dump()
    }