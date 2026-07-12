"""
Groq API 채팅 응답 생성 (무료 고속 LLM)
"""
import os
from groq import Groq

def generate_chat_response(messages, system_prompt=""):
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    chat_messages = []
    if system_prompt:
        chat_messages.append({"role": "system", "content": system_prompt})
    for msg in messages:
        if isinstance(msg, dict) and "role" in msg and "content" in msg:
            chat_messages.append({"role": msg["role"], "content": msg["content"]})

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=chat_messages,
        max_tokens=600,
        temperature=0.7
    )
    return response.choices[0].message.content

def generate_pose_feedback(pose_summary):
    try:
        client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "당신은 사격 자세 전문 코치입니다."},
                {"role": "user", "content": f"다음 자세 데이터를 분석해주세요:\n{pose_summary}"}
            ],
            max_tokens=300
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"피드백 생성 실패: {str(e)}"

def generate_basic_feedback(pose_data):
    return "자세 분석 완료"
