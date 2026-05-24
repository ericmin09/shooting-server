"""
Shooting form feedback generator using local LLM (no OpenAI rate limits).
Uses gpt4all for offline inference.
"""

import json
from local_llm import generate_pose_feedback, generate_basic_feedback

def generate_feedback(pose_data, image_path=None):
    """
    포즈 데이터를 기반으로 사격 자세 피드백 생성 (로컬 LLM 사용, 레이트 제한 없음)
    
    Args:
        pose_data: MediaPipe 포즈 분석 데이터
        image_path: 분석할 이미지 파일 경로 (선택사항)
    
    Returns:
        사격 자세 평가 및 위험 요소 분석 결과
    """
    
    # 에러 처리
    if pose_data is None:
        return "포즈 분석 실패"
    
    if isinstance(pose_data, dict) and "error" in pose_data:
        return pose_data["error"]
    
    if isinstance(pose_data, dict) and pose_data.get("status") != "success":
        return "포즈 분석에 실패했습니다"
    
    try:
        # 포즈 데이터 문자열화
        pose_summary = format_pose_data(pose_data)
        
        # 로컬 LLM으로 피드백 생성 (레이트 제한 없음)
        result = generate_pose_feedback(pose_summary)
        
        # 결과 포맷팅
        return format_gpt_feedback(result)
        
    except Exception as e:
        print(f"피드백 생성 실패: {e}")
        # 실패 시 기본 피드백 반환
        return generate_basic_feedback_text(pose_data)


def format_pose_data(pose_data):
    """포즈 데이터를 읽기 쉬운 형식으로 변환"""
    if not isinstance(pose_data, dict):
        return str(pose_data)
    
    summary = []
    
    if "posture_grade" in pose_data:
        summary.append(f"자세 등급: {pose_data['posture_grade']}")
    
    if "overall_posture_score" in pose_data:
        summary.append(f"전체 점수: {pose_data['overall_posture_score']}/100")
    
    if "left_arm_angle" in pose_data:
        summary.append(f"왼팔 각도: {pose_data['left_arm_angle']}°")
    
    if "right_arm_angle" in pose_data:
        summary.append(f"오른팔 각도: {pose_data['right_arm_angle']}°")
    
    if "shoulder_balance_px" in pose_data:
        summary.append(f"어깨 균형: {pose_data['shoulder_balance_px']}px 차이")
    
    if "spine_tilt_px" in pose_data:
        summary.append(f"척추 기울임: {pose_data['spine_tilt_px']}px")
    
    if "hip_balance_px" in pose_data:
        summary.append(f"골반 균형: {pose_data['hip_balance_px']}px 차이")
    
    if "posture_details" in pose_data and isinstance(pose_data["posture_details"], list):
        summary.append("상세 평가:")
        for detail in pose_data["posture_details"]:
            summary.append(f"  - {detail}")
    
    return "\n".join(summary)

def format_gpt_feedback(gpt_result):
    """로컬 LLM 응답을 사용자 친화적 형식으로 변환"""
    if not isinstance(gpt_result, dict):
        return str(gpt_result)
    
    feedback = []
    
    # 자세 점수
    if "posture_score" in gpt_result:
        score = gpt_result['posture_score']
        if isinstance(score, str):
            try:
                score = int(score)
            except:
                score = 50
        feedback.append(f"🎯 자세 안정성: {score}/100")
    
    # 전체 평가
    if "overall_assessment" in gpt_result:
        feedback.append(f"\n📊 전체 평가:\n{gpt_result['overall_assessment']}")
    
    # 개선사항
    if "improvements" in gpt_result and isinstance(gpt_result["improvements"], list):
        improvements = [i for i in gpt_result["improvements"] if i]
        if improvements:
            feedback.append("\n⚙️ 개선 사항:")
            for improvement in improvements:
                feedback.append(f"  • {improvement}")
    
    # 안전 우려사항 (중요)
    if "safety_concerns" in gpt_result and isinstance(gpt_result["safety_concerns"], list):
        safety_items = [item for item in gpt_result["safety_concerns"] if item]
        if safety_items:
            feedback.append("\n⚠️ 안전 우려사항:")
            for concern in safety_items:
                feedback.append(f"  🔴 {concern}")
    
    # 권장사항
    if "recommendations" in gpt_result:
        recs = gpt_result["recommendations"]
        if isinstance(recs, str) and recs:
            feedback.append(f"\n✅ 권장사항:\n  ✓ {recs}")
        elif isinstance(recs, list):
            recommendations = [r for r in recs if r]
            if recommendations:
                feedback.append("\n✅ 권장사항:")
                for rec in recommendations:
                    feedback.append(f"  ✓ {rec}")
    
    # 사격 준비 상태
    if "shooting_ready" in gpt_result:
        ready = gpt_result["shooting_ready"]
        status = "✅ 준비완료" if ready else "❌ 자세 조정 필요"
        feedback.append(f"\n🎯 사격 준비 상태: {status}")
    
    return "\n".join(feedback) if feedback else "분석 완료"

def generate_basic_feedback_text(pose_data):
    """로컬 LLM 없이 기본 피드백 생성"""
    if not isinstance(pose_data, dict):
        return "피드백을 생성할 수 없습니다"
    
    feedback = []
    
    # 자세 등급
    if "posture_grade" in pose_data:
        grade = pose_data["posture_grade"]
        grade_feedback = {
            "A": "✅ 우수한 자세입니다!",
            "B": "⚠️ 양호하지만 개선 필요",
            "C": "❌ 조정 필요",
            "D": "❌ 심각한 문제"
        }
        feedback.append(grade_feedback.get(grade, f"자세 등급: {grade}"))
    
    # 점수 기반 피드백
    if "overall_posture_score" in pose_data:
        score = pose_data["overall_posture_score"]
        if score >= 80:
            feedback.append("자세가 매우 좋습니다!")
        elif score >= 60:
            feedback.append("자세가 괜찮지만 약간의 조정이 필요합니다.")
        else:
            feedback.append("자세를 크게 조정해야 합니다.")
    
    # 각도 기반 경고
    left_angle = pose_data.get("left_arm_angle", 180)
    right_angle = pose_data.get("right_arm_angle", 180)
    
    if left_angle < 120 or right_angle < 120:
        feedback.append("⚠️ 팔이 과도하게 구부러져 있습니다.")
    
    shoulder_balance = pose_data.get("shoulder_balance_px", 0)
    if abs(shoulder_balance) > 30:
        feedback.append("⚠️ 어깨가 비뚤어져 있습니다.")
    
    spine_tilt = pose_data.get("spine_tilt_px", 0)
    if abs(spine_tilt) > 40:
        feedback.append("⚠️ 척추가 과도하게 기울어져 있습니다.")
    
    return "\n".join(feedback) if feedback else "분석 완료"
