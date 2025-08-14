import os
from typing import Dict, Any
from google import genai

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# 모델 예: gemini-2.5-flash, gemini-2.0-pro 등(계정 보유 모델만 동작)

def run_gemini(model: str, messages, temperature: float, max_tokens: int | None) -> Dict[str, Any]:
    # 간단히 user/assistant 내용을 붙여 하나의 프롬프트로 보냄(기본 버전)
    parts = []
    system_texts = []
    for m in messages:
        if m["role"] == "system":
            system_texts.append(m["content"])
        else:
            parts.append(f"{m['role']}: {m['content']}")

    prompt = ("\n".join(system_texts) + "\n\n" if system_texts else "") + "\n".join(parts)

    resp = client.models.generate_content(
        model=model,
        contents=prompt,
        config={
            "temperature": temperature,
            "max_output_tokens": max_tokens or 1024,
        },
    )

    text = getattr(resp, "text", None) or ""  # 간단 추출
    # google-genai 에서는 usage 요약이 응답 객체 메타에 포함될 수 있음(간단화)
    return {"output_text": text, "usage": None, "raw": resp.to_dict() if hasattr(resp, "to_dict") else None}