import os
from typing import Dict, Any, List
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def _to_text_pair(messages: List[dict]):
    system_text = "\n".join([m["content"] for m in messages if m["role"] == "system"]) or None
    user_parts = []
    for m in messages:
        if m["role"] in ("user", "assistant"):
            prefix = "User:" if m["role"] == "user" else "Assistant:"
            user_parts.append(f"{prefix} {m['content']}")
    input_text = "\n".join(user_parts)
    return system_text, input_text

def run_openai(model: str, messages, temperature: float, max_tokens: int | None) -> Dict[str, Any]:
    system_text, input_text = _to_text_pair(messages)

    # 신버전(>=1.x) : responses API가 있는 경우
    if hasattr(client, "responses"):
        resp = client.responses.create(
            model=model,
            instructions=system_text,
            input=input_text,
            temperature=temperature,
            max_output_tokens=max_tokens,
        )
        return {
            "output_text": getattr(resp, "output_text", "")
                          or (resp.output[0].content[0].text if getattr(resp, "output", None) else ""),
            "usage": getattr(resp, "usage", None).model_dump() if hasattr(resp, "usage") else None,
            "raw": resp.model_dump() if hasattr(resp, "model_dump") else None,
        }

    # 구버전(0.x) : chat.completions 로 폴백
    chat_msgs = []
    if system_text:
        chat_msgs.append({"role": "system", "content": system_text})
    # user/assistant 원문 그대로 전달
    for m in messages:
        if m["role"] in ("user", "assistant"):
            chat_msgs.append({"role": m["role"], "content": m["content"]})

    # 구버전 클라이언트는 client.chat.completions.create 사용
    resp = client.chat.completions.create(
        model=model,
        messages=chat_msgs,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    text = resp.choices[0].message.content if resp.choices else ""
    usage = getattr(resp, "usage", None)
    return {"output_text": text, "usage": usage, "raw": resp.dict() if hasattr(resp, "dict") else None}
