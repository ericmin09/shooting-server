"""
OpenAI API 채팅 응답 생성 - 재시도 로직 포함
"""
import os
import time
import openai

def generate_chat_response(messages, system_prompt=""):
    client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    chat_messages = []
    if system_prompt:
        chat_messages.append({"role": "system", "content": system_prompt})
    for msg in messages:
        if isinstance(msg, dict) and "role" in msg and "content" in msg:
            chat_messages.append({"role": msg["role"], "content": msg["content"]})

    # 429 Rate Limit 시 최대 3번 재시도 (2초, 4초 대기)
    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=chat_messages,
                max_tokens=600,
                temperature=0.7
            )
            return response.choices[0].message.content
        except openai.RateLimitError:
            if attempt < 2:
                wait_seconds = (attempt + 1) * 2
                print(f"[Chat] 429 Rate Limit - {wait_seconds}초 후 재시도 ({attempt+1}/3)")
                time.sleep(wait_seconds)
            else:
                raise openai.RateLimitError("요청이 너무 많습니다. 잠시 후 다시 시도해주세요.", response=None, body=None)
        except Exception:
            raise

def generate_pose_feedback(pose_summary):
    try:
        client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        response = client.chat.completions.create(
            model="gpt-4o-mini",
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
