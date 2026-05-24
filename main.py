from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
import shutil
import os
import time
import sys
from PIL import Image
from dotenv import load_dotenv
from pydantic import BaseModel
import hashlib
import cloudinary
import cloudinary.uploader

# .env 파일 로드
load_dotenv()

# Cloudinary 설정
cloudinary.config(
    cloud_name=os.getenv("CLOUDINARY_CLOUD_NAME"),
    api_key=os.getenv("CLOUDINARY_API_KEY"),
    api_secret=os.getenv("CLOUDINARY_API_SECRET")
)

from pose_analyzer import analyze_pose
from feedback import generate_feedback

# 응답 캐시 (메모리 캐시)
response_cache = {}



# 스크립트가 있는 디렉토리를 기준으로 설정 (가장 안전한 방법)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOADS_DIR = os.path.join(SCRIPT_DIR, "uploads")
# uploads/uploads 방지
if UPLOADS_DIR.endswith("uploads/uploads"):
    UPLOADS_DIR = os.path.join(SCRIPT_DIR, "uploads")

app = FastAPI()

# 채팅 메시지 모델
class ChatMessage(BaseModel):
    message: str

class ChatRequest(BaseModel):
    messages: list  # 채팅 히스토리

# /images 경로로 업로드 파일 접근 가능하게 마운트
app.mount("/images", StaticFiles(directory=UPLOADS_DIR), name="images")

print(f"Script directory: {SCRIPT_DIR}")
print(f"Uploads directory: {UPLOADS_DIR}")

# uploads 폴더 생성
try:
    os.makedirs(UPLOADS_DIR, exist_ok=True)
    print(f"✓ Uploads directory created/verified")
except Exception as e:
    print(f"✗ Error with uploads directory: {e}")
    sys.exit(1)

@app.get("/")
def root():
    return {"message": "server is running"}

@app.post("/upload")
async def upload_image(file: UploadFile = File(...)):
    try:
        # 파일 타입 기본 검증
        if not file.content_type or not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="이미지 파일만 업로드 가능합니다")
        
        # 파일명 생성
        filename = f"{int(time.time())}_{file.filename}"
        filepath = os.path.join(UPLOADS_DIR, filename)
        print(f"\n{'='*60}")
        print(f"[파일 업로드]")
        print(f"파일명: {filename}")
        print(f"저장 경로(UPLOADS_DIR): {UPLOADS_DIR}")
        print(f"실제 저장 경로: {filepath}")
        
        # 디렉토리 확인
        if not os.path.exists(UPLOADS_DIR):
            print(f"WARNING: uploads 디렉토리 없음, 생성 중...")
            os.makedirs(UPLOADS_DIR, exist_ok=True)
        
        # 파일 저장
        print(f"파일 저장 중...")
        with open(filepath, "wb") as f:
            contents = await file.read()
            f.write(contents)
        
        print(f"파일이 저장됨: {filepath}")
        
        # 파일 저장 확인
        if not os.path.exists(filepath):
            error_msg = f"파일 저장 실패: {filepath}"
            print(f"ERROR: {error_msg}")
            return {"status": "failed", "error": error_msg}
        
        # 파일 크기 확인
        file_size = os.path.getsize(filepath)
        print(f"파일 크기: {file_size} bytes")
        
        if file_size == 0:
            error_msg = "파일이 비어있습니다"
            print(f"ERROR: {error_msg}")
            return {"status": "failed", "error": error_msg}
        
        # 파일 타입 상세 검증
        try:
            with Image.open(filepath) as img:
                file_type = img.format.lower()
                if file_type not in ['jpeg', 'jpg', 'png', 'bmp', 'tiff', 'gif']:
                    os.remove(filepath)  # 잘못된 파일 삭제
                    error_msg = f"지원되지 않는 이미지 형식입니다: {file_type}"
                    print(f"ERROR: {error_msg}")
                    return {"status": "failed", "error": error_msg}
        except Exception as e:
            os.remove(filepath)  # 잘못된 파일 삭제
            error_msg = f"이미지 파일이 아닙니다: {str(e)}"
            print(f"ERROR: {error_msg}")
            return {"status": "failed", "error": error_msg}
        
        print(f"✓ 파일 타입 확인: {file_type}")
        
        # 포즈 분석
        print(f"포즈 분석 시작...")
        pose_data = analyze_pose(filepath)
        
        print(f"pose_data type: {type(pose_data)}")
        print(f"pose_data keys: {pose_data.keys() if isinstance(pose_data, dict) else 'not dict'}")
        
        # numpy 타입을 JSON 직렬화 가능한 타입으로 변환
        import numpy as np
        def convert_numpy_types(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, (np.int32, np.int64)):
                return int(obj)
            elif isinstance(obj, dict):
                return {k: convert_numpy_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            else:
                return obj
        
        pose_data = convert_numpy_types(pose_data)
        
        if pose_data is None:
            error_msg = "포즈 분석 실패"
            print(f"ERROR: {error_msg}")
            return {"status": "failed", "error": error_msg}
        
        print(f"완료!")
        print(f"{'='*60}\n")
        
        # 평가 기준 설명 추가
        criteria_descriptions = {
            "body_lean": {
                "name": "A. 상체 기울기 (Body Lean)",
                "max_score": 20,
                "description": "어깨와 엉덩이를 이은 선의 각도를 계산합니다. 전방 5~15도가 이상적입니다."
            },
            "arm_extension": {
                "name": "B. 팔 각도 (Arm Extension)",
                "max_score": 25,
                "description": "팔이 펼쳐진 정도를 측정합니다. 160~180도가 좋은 범위입니다."
            },
            "shoulder_level": {
                "name": "C. 어깨 수평 (Shoulder Level)",
                "max_score": 15,
                "description": "좌우 어깨의 높이 차이를 계산합니다. 작을수록 좋습니다."
            },
            "head_position": {
                "name": "D. 머리 위치 (Head Position)",
                "max_score": 15,
                "description": "머리가 어깨 중심에서 얼마나 벗어나는지를 측정합니다."
            },
            "arm_wrist_alignment": {
                "name": "E. 팔/손 정렬 (Arm/Wrist Alignment)",
                "max_score": 25,
                "description": "어깨-팔꿈치-손목이 일직선에 가까운지를 평가합니다."
            }
        }
        
        # 평가 항목별 상세 정보 구성
        scores = pose_data.get("scores", {})
        
        # 각 항목 점수 추출
        body_lean = scores.get("body_lean", {})
        arm_ext = scores.get("arm_extension", {})
        shoulder = scores.get("shoulder_level", {})
        head = scores.get("head_position", {})
        alignment = scores.get("arm_wrist_alignment", {})
        
        # 점수 합계 계산
        score_a = body_lean.get('score', 0)
        score_b = arm_ext.get('score', 0)
        score_c = shoulder.get('score', 0)
        score_d = head.get('score', 0)
        score_e = alignment.get('score', 0)
        
        total_calculated = score_a + score_b + score_c + score_d + score_e
        
        # 점수 요약 문자열
        evaluation_details = []
        
        # 최종 등급과 점수 첫 줄 강조
        grade = pose_data.get('posture_grade', 'N/A')
        grade_map = {
            'A': '우수한 자세 (Excellent)',
            'B': '좋은 자세 (Good)',
            'C': '보통 자세 (Fair)',
            'D': '개선 필요 (Needs Improvement)',
            'F': '심각한 문제 (Poor)'
        }
        grade_description = grade_map.get(grade, '')
        
        evaluation_details.append(f"═══════════════════════════════════")
        evaluation_details.append(f"최종 자세 등급: [{grade}]")
        evaluation_details.append(f"등급명: {grade_description}")
        evaluation_details.append(f"총점: {pose_data.get('overall_score', 0)}/100점")
        evaluation_details.append(f"═══════════════════════════════════")
        evaluation_details.append("")
        
        # 점수 구성 (한 줄로 명확하게)
        evaluation_details.append(f"점수 계산: {score_a}+{score_b}+{score_c}+{score_d}+{score_e} = {total_calculated}점")
        evaluation_details.append("")
        evaluation_details.append("【 세부 평가 항목 】")
        evaluation_details.append("")
        
        # A. 상체 기울기
        evaluation_details.append(f"[A] 상체 기울기 {score_a}/20점")
        evaluation_details.append(f"    → {body_lean.get('eval', 'N/A')}")
        evaluation_details.append("")
        
        # B. 팔 각도
        evaluation_details.append(f"[B] 팔 각도 {score_b}/25점")
        evaluation_details.append(f"    → {arm_ext.get('eval', 'N/A')}")
        evaluation_details.append("")
        
        # C. 어깨 수평
        evaluation_details.append(f"[C] 어깨 수평 {score_c}/15점")
        evaluation_details.append(f"    → {shoulder.get('eval', 'N/A')}")
        evaluation_details.append("")
        
        # D. 머리 위치
        evaluation_details.append(f"[D] 머리 위치 {score_d}/15점")
        evaluation_details.append(f"    → {head.get('eval', 'N/A')}")
        evaluation_details.append("")
        
        # E. 팔/손 정렬
        evaluation_details.append(f"[E] 팔/손 정렬 {score_e}/25점")
        evaluation_details.append(f"    → {alignment.get('eval', 'N/A')}")
        evaluation_details.append("")
        
        evaluation_text = "\n".join(evaluation_details)
        
        # 피드백 리스트를 간단하게 정리 (이모지 제거, 간결하게)
        feedback_list = pose_data.get("feedback", [])
        simplified_feedback = []
        
        for fb in feedback_list:
            # 이모지 제거
            clean_fb = fb.replace('✓', '✓').replace('⚠️', '⚠').replace('❌', '✗').replace('🎯', '•')
            simplified_feedback.append(clean_fb)
        
        feedback_text = "\n".join([f"{item}" for item in simplified_feedback]) if simplified_feedback else "피드백이 없습니다."
        
        # 시각화 이미지 URL (Cloudinary로 업로드)
        visualization_url = None
        if "visualization_image" in pose_data and pose_data["visualization_image"]:
            vis_filename = pose_data['visualization_image']
            vis_path = os.path.join(UPLOADS_DIR, vis_filename)
            
            if os.path.exists(vis_path):
                try:
                    # Cloudinary에 업로드
                    upload_result = cloudinary.uploader.upload(
                        vis_path,
                        folder="shootinganal/landmarks",
                        resource_type="auto",
                        public_id=os.path.splitext(vis_filename)[0]
                    )
                    visualization_url = upload_result.get("secure_url")
                    print(f"✓ 시각화 이미지 Cloudinary 업로드 완료: {visualization_url}")
                except Exception as e:
                    print(f"⚠ Cloudinary 업로드 실패: {e}")
                    visualization_url = None
            else:
                print(f"⚠ 시각화 이미지 파일 없음: {vis_path}")
        else:
            print(f"⚠ 시각화 이미지 생성 안됨")
        
        # 최종 응답 구조 (관절포인트가 그려진 이미지만 반환)
        response_data = {
            "status": "success",
            
            # 핵심 정보 (앱에서 바로 표시)
            "posture_grade": pose_data.get("posture_grade", "N/A"),
            "overall_score": pose_data.get("overall_score", 0),
            "grade_text": pose_data.get("grade_text", ""),
            
            # 상세 평가 (텍스트 형태로 바로 표시 가능)
            "evaluation_details": evaluation_text,
            "feedback": feedback_text,
            
            # 관절포인트가 그려진 이미지만
            "visualization_url": visualization_url,
            
            # 원본 데이터 (필요시 참고)
            "raw_scores": scores
        }
        
        print(f"응답 구조:")
        print(f"  - overall_score: {response_data['overall_score']}")
        print(f"  - posture_grade: {response_data['posture_grade']}")
        print(f"  - evaluation_details: {len(evaluation_details)} 줄")
        print(f"  - feedback: {len(feedback_list)} 항목")
        
        return response_data
    
    except HTTPException:
        raise
    except Exception as e:
        error_msg = f"{type(e).__name__}: {str(e)}"
        print(f"ERROR: {error_msg}")
        import traceback
        traceback.print_exc()
        return {
            "status": "error",
            "error": error_msg
        }

@app.post("/chat")
async def chat(request: ChatRequest):
    """
    사격에 대해 물어보는 채팅 엔드포인트 (로컬 LLM 사용, 레이트 제한 없음)
    
    Args:
        request: ChatRequest 객체 (messages: 채팅 히스토리 목록)
    
    Returns:
        로컬 LLM의 응답 메시지
    """
    try:
        from local_llm import generate_chat_response
        
        system_prompt = """당신은 전문적인 사격 자세 및 안전 코치입니다. 
사격에 관한 모든 질문에 대해 친절하고 자세하게 답변해주세요.
특히 안전, 자세, 균형, 팔 각도, 어깨 위치, 머리 위치, 호흡 등에 대해 전문 지식을 갖추고 있습니다.
한국어로 명확하고 쉽게 설명해주세요."""
        
        # 로컬 LLM으로 응답 생성 (레이트 제한 없음, 약 10-30초)
        reply_text = generate_chat_response(
            messages=request.messages if isinstance(request.messages, list) else [],
            system_prompt=system_prompt
        )
        
        return {
            "status": "success",
            "message": reply_text
        }
        
    except Exception as e:
        print(f"Chat error: {e}")
        import traceback
        traceback.print_exc()
        return {
            "status": "error",
            "message": f"채팅 오류: {str(e)}"
        }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)