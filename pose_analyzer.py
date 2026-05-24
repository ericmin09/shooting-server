import cv2
import mediapipe as mp
import numpy as np
import os
from PIL import Image as PILImage
import math

# MediaPipe 포즈 초기화 (새로운 API)
mp_pose = mp.tasks.vision.PoseLandmarker
mp_drawing = mp.tasks.vision.drawing_utils
mp_drawing_styles = mp.tasks.vision.drawing_styles

# 글로벌 포즈 모델 캐시 (한 번만 초기화)
_pose_model = None

def get_pose_model():
    """MediaPipe 포즈 모델 초기화 (새로운 API)"""
    global _pose_model
    if _pose_model is None:
        print("MediaPipe 포즈 모델 초기화 중...")
        # 다운로드된 모델 파일 사용
        model_path = os.path.join(os.path.dirname(__file__), "pose_landmarker.task")
        if not os.path.exists(model_path):
            # 모델이 없으면 다운로드 (SSL 검증 우회)
            import urllib.request
            import ssl
            ssl._create_default_https_context = ssl._create_unverified_context
            model_url = 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task'
            print("포즈 모델 다운로드 중...")
            try:
                urllib.request.urlretrieve(model_url, model_path)
                print("✓ 모델 다운로드 완료")
            except Exception as e:
                print(f"✗ 모델 다운로드 실패: {e}")
                raise e

        base_options = mp.tasks.BaseOptions(model_asset_path=model_path)
        options = mp.tasks.vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
            min_pose_detection_confidence=0.3,
            min_pose_presence_confidence=0.3,
            min_tracking_confidence=0.3,
            output_segmentation_masks=False
        )
        _pose_model = mp.tasks.vision.PoseLandmarker.create_from_options(options)
        print("✓ MediaPipe 포즈 모델 초기화 완료")
    return _pose_model

def calculate_angle(a, b, c):
    """세 점 사이의 각도 계산"""
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)

    if angle > 180:
        angle = 360-angle

    return angle

def enhance_image(image):
    """이미지 전처리: 관절 포인트 인식을 위한 강화 버전"""
    print("이미지 전처리 중 (관절 포인트 최적화)...")
    
    # 1. 원본 이미지 백업
    original = image.copy()
    
    # 2. 밝기 및 콘트라스트 조정 (LAB 색상 공간)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # CLAHE 강화 설정 - 관절 포인트 선명도 향상
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(12, 12))
    l = clahe.apply(l)
    
    # 합치기
    enhanced = cv2.merge([l, a, b])
    enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
    
    # 3. 노이즈 제거 (관절 포인트 보존)
    enhanced = cv2.bilateralFilter(enhanced, 7, 50, 50)
    
    # 4. 언샤프 마스킹 강화 (관절 엣지 강조)
    gaussian = cv2.GaussianBlur(enhanced, (0, 0), 1.5)
    enhanced = cv2.addWeighted(enhanced, 2.2, gaussian, -1.2, 0)
    
    # 5. 엣지 강화 (관절 실루엣 강조)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    enhanced = cv2.morphologyEx(enhanced, cv2.MORPH_CLOSE, kernel)
    
    # 6. 대비도 강화 (관절 포인트 구분력 향상)
    enhanced = cv2.convertScaleAbs(enhanced, alpha=1.1, beta=5)
    
    return enhanced

def safe_get_point(keypoints, idx):
    """안전한 포인트 추출"""
    if idx < keypoints.shape[0]:
        point = keypoints[idx]
        if point[0] > 0 and point[1] > 0:
            return point
    return None

def generate_shooting_feedback(results_dict):
    """
    사격 자세 분석 결과를 바탕으로 사용자 친화적인 피드백 생성
    """
    feedback = []
    scores = results_dict.get("scores", {})
    total_score = results_dict.get("overall_score", 0)
    grade = results_dict.get("posture_grade", "N/A")
    grade_text = results_dict.get("grade_text", "")
    
    # 등급 정보를 첫 줄에 추가
    feedback.append(f"【 자세 등급: {grade} 】 {grade_text}")
    feedback.append("")
    
    # A. 상체 기울기 피드백
    body_lean = scores.get("body_lean", {})
    body_lean_score = body_lean.get("score", 0)
    
    if body_lean_score >= 18:
        feedback.append("✓ 상체 기울기가 완벽합니다!")
    elif body_lean_score >= 14:
        feedback.append("⚠️ 상체 기울기를 조정해주세요 - " + body_lean.get("eval", ""))
    else:
        feedback.append("❌ " + body_lean.get("eval", ""))

    # B. 팔 각도 피드백
    arm_ext = scores.get("arm_extension", {})
    arm_score = arm_ext.get("score", 0)
    
    if arm_score >= 23:
        feedback.append("✓ 팔이 완벽하게 펼쳐져 있습니다!")
    elif arm_score >= 20:
        feedback.append("⚠️ 팔을 조금 더 펼쳐주세요")
    elif arm_score >= 15:
        feedback.append("⚠️ 팔을 더 펼쳐주세요")
    else:
        feedback.append("❌ 팔을 훨씬 더 펼쳐주세요 - " + arm_ext.get("eval", ""))

    # C. 어깨 수평 피드백
    shoulder = scores.get("shoulder_level", {})
    shoulder_score = shoulder.get("score", 0)
    
    if shoulder_score >= 14:
        feedback.append("✓ 어깨가 완벽하게 수평입니다!")
    elif shoulder_score >= 12:
        feedback.append("⚠️ 어깨를 수평으로 맞춰주세요 - 약간 기울어져 있습니다")
    elif shoulder_score >= 10:
        feedback.append("⚠️ 어깨가 기울어져 있습니다 - 수평을 맞춰주세요")
    else:
        feedback.append("❌ 어깨를 크게 기울여졌습니다 - 반드시 수평으로 맞춰주세요")

    # D. 머리 위치 피드백
    head = scores.get("head_position", {})
    head_score = head.get("score", 0)
    
    if head_score >= 14:
        feedback.append("✓ 머리 위치가 좋습니다!")
    elif head_score >= 12:
        feedback.append("⚠️ 머리 위치를 조정해주세요 - 약간 벗어나가 있습니다")
    elif head_score >= 10:
        feedback.append("⚠️ 머리 위치를 중심에 맞춰주세요")
    else:
        feedback.append("❌ 머리 위치가 크게 벗어나 있습니다")

    # E. 팔/손 정렬 피드백
    alignment = scores.get("arm_wrist_alignment", {})
    align_score = alignment.get("score", 0)
    
    if align_score >= 24:
        feedback.append("✓ 팔과 손목이 완벽하게 정렬되어 있습니다!")
    elif align_score >= 22:
        feedback.append("⚠️ 팔 정렬을 조정해주세요 - 좋으나 약간 개선 필요")
    elif align_score >= 20:
        feedback.append("⚠️ 팔과 손목을 더 일직선에 맞춰주세요")
    elif align_score >= 16:
        feedback.append("⚠️ 팔이 기울어져 있습니다 - 일직선에 맞춰주세요")
    else:
        feedback.append("❌ 팔을 펴고 손목을 일직선에 맞춰주세요")

    # 총합 피드백
    feedback.append("")  # 빈줄
    if total_score >= 90:
        feedback.append("🎯 종합 의견: 우수한 자세입니다! 현재 상태를 유지하세요.")
    elif total_score >= 80:
        feedback.append("🎯 종합 의견: 좋은 자세입니다. 작은 조정만으로 완벽해질 겁니다.")
    elif total_score >= 70:
        feedback.append("🎯 종합 의견: 기본은 잡았습니다. 위의 항목들을 개선해주세요.")
    elif total_score >= 60:
        feedback.append("🎯 종합 의견: 많은 개선이 필요합니다. 각 항목의 피드백을 참고하세요.")
    else:
        feedback.append("🎯 종합 의견: 자세를 완전히 다시 조정해주세요. 모든 항목을 확인해주세요.")

    return feedback


def draw_pose_landmarks(image, landmarks):
    """
    MediaPipe 포즈 랜드마크를 이미지 위에 그려서 시각화
    """
    vis_image = image.copy()
    
    # MediaPipe 포즈 연결선 정의 (33개 랜드마크)
    connections = [
        # 얼굴
        (0, 1), (1, 2), (2, 3), (3, 7),
        (0, 4), (4, 5), (5, 6), (6, 8),
        (9, 10),
        
        # 몸통
        (11, 12),  # 어깨 연결
        (11, 23), (12, 24),  # 어깨-엉덩이
        (23, 24),  # 엉덩이 연결
        
        # 왼쪽 팔
        (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
        
        # 오른쪽 팔
        (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
        
        # 왼쪽 다리
        (23, 25), (25, 27), (27, 29), (27, 31), (29, 31),
        
        # 오른쪽 다리
        (24, 26), (26, 28), (28, 30), (28, 32), (30, 32),
        
        # 발
        (27, 29), (28, 30), (29, 31), (30, 32)
    ]
    
    # 랜드마크를 픽셀 좌표로 변환
    height, width, _ = vis_image.shape
    landmark_points = []
    
    for landmark in landmarks:
        x = int(landmark.x * width)
        y = int(landmark.y * height)
        landmark_points.append((x, y))
    
    # 연결선 그리기
    for connection in connections:
        start_idx, end_idx = connection
        if start_idx < len(landmark_points) and end_idx < len(landmark_points):
            start_point = landmark_points[start_idx]
            end_point = landmark_points[end_idx]
            
            # 연결선 그리기 (흰색)
            cv2.line(vis_image, start_point, end_point, (255, 255, 255), 3)
            cv2.line(vis_image, start_point, end_point, (0, 255, 0), 2)
    
    # 각 포인트 그리기
    for i, point in enumerate(landmark_points):
        # 포인트 타입에 따라 색상 구분
        if i in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:  # 얼굴
            color = (255, 255, 0)  # 시안
        elif i in [11, 12]:  # 어깨
            color = (0, 0, 255)  # 빨강
        elif i in [13, 14]:  # 팔꿈치
            color = (0, 165, 255)  # 오렌지
        elif i in [15, 16]:  # 손목
            color = (0, 255, 255)  # 노랑
        elif i in [17, 18, 19, 20, 21, 22]:  # 손가락
            color = (255, 0, 255)  # 마젠타
        elif i in [23, 24]:  # 엉덩이
            color = (255, 0, 0)  # 파랑
        elif i in [25, 26]:  # 무릎
            color = (255, 0, 127)  # 핑크
        elif i in [27, 28, 29, 30, 31, 32]:  # 발목/발
            color = (127, 0, 255)  # 보라
        else:
            color = (255, 255, 255)  # 흰색
        
        # 포인트 원 그리기
        cv2.circle(vis_image, point, 8, color, -1)  # 채워진 원
        cv2.circle(vis_image, point, 12, (255, 255, 255), 2)  # 흰색 테두리
        
        # 라벨 표시 (작게)
        cv2.putText(vis_image, str(i), (point[0] + 15, point[1] - 15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    # 감지된 포인트 수 표시
    detected_points = len(landmark_points)
    cv2.putText(vis_image, f"Detected: {detected_points}/33 landmarks", (20, 40), 
               cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
    
    # 범례 추가
    legend_y = 80
    legends = [
        ("Face", (255, 255, 0)),
        ("Shoulders", (0, 0, 255)),
        ("Elbows", (0, 165, 255)),
        ("Wrists", (0, 255, 255)),
        ("Fingers", (255, 0, 255)),
        ("Hips", (255, 0, 0)),
        ("Knees", (255, 0, 127)),
        ("Feet", (127, 0, 255))
    ]
    
    for label, color in legends:
        cv2.putText(vis_image, label, (20, legend_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        legend_y += 30
    
    return vis_image

def evaluate_body_lean(shoulder_center, hip_center):
    """
    A. 상체 기울기 (Body Lean) - 20점
    어깨 중심과 엉덩이 중심을 이은 선의 각도 계산
    
    좌표계 정의: 
    - y축이 아래쪽으로 증가하는 이미지 좌표계 기준
    - 양수 각도 = 상체가 앞으로 기울어짐 (forward lean)
    - 음수 각도 = 상체가 뒤로 기울어짐 (backward lean)
    
    좋은 범위: 5도 ~ 15도의 전방 기울기
    """
    if shoulder_center is None or hip_center is None:
        return {"score": 0, "eval": "계산불가", "detail": "어깨 또는 엉덩이 감지 불가", "angle": 0}
    
    # 벡터: hip_center -> shoulder_center (위쪽이 음수)
    dx = shoulder_center[0] - hip_center[0]
    dy = shoulder_center[1] - hip_center[1]  # 음수 = 위쪽
    
    # 각도 계산: 수직선(아래쪽)을 기준으로 시계방향
    # 각도가 양수 = 앞으로 기울어짐
    import math
    body_lean_angle = math.degrees(math.atan2(dx, -dy))  # -dy: 위쪽 방향을 0도로
    
    # forward lean 이므로 절댓값 사용
    lean_angle = body_lean_angle
    
    print(f"[DEBUG] 상체 기울기: {lean_angle:.1f}° (어깨: {shoulder_center}, 엉덩이: {hip_center})")
    
    # 패널티: 너무 수직이거나 뒤로 기울면 감점
    if 5 <= lean_angle <= 15:
        score = 20  # 완벽
        eval_text = "완벽한 전방 기울기"
    elif 3 <= lean_angle < 5:
        score = 18
        eval_text = "좋음 (약간 덜 기울음)"
    elif 15 < lean_angle <= 20:
        score = 17
        eval_text = "좋음 (약간 더 기울음)"
    elif 0 <= lean_angle < 3:
        score = 14
        eval_text = "거의 수직 (조금 더 기울여주세요)"
    elif 20 < lean_angle <= 25:
        score = 14
        eval_text = "과도하게 기울어짐"
    elif lean_angle < 0:
        score = 10
        eval_text = "뒤로 기울어짐 (상체를 앞으로 기울여주세요)"
    else:
        score = 8
        eval_text = "매우 과도하게 기울어짐"
    
    return {
        "score": score,
        "eval": eval_text,
        "detail": f"기울기 각도: {lean_angle:.1f}°",
        "angle": round(lean_angle, 1)
    }


def evaluate_arm_extension(left_arm_angle, right_arm_angle):
    """
    B. 팔 각도 (Arm Extension) - 25점
    어깨-팔꿈치-손목 3점 각도 계산
    160도 ~ 180도 = good
    140도 이하 = bad
    """
    angles = []
    if left_arm_angle is not None:
        angles.append(left_arm_angle)
    if right_arm_angle is not None:
        angles.append(right_arm_angle)
    
    if not angles:
        return {"score": 0, "eval": "계산불가", "detail": "팔 감지 불가", "average_angle": 0}
    
    avg_angle = sum(angles) / len(angles)
    
    # 게이지 계산
    if avg_angle >= 160:
        if avg_angle >= 175:
            score = 25
            eval_text = "완벽하게 펼쳐짐"
        elif avg_angle >= 170:
            score = 24
            eval_text = "우수 (거의 완벽)"
        else:  # 160 ~ 170
            score = 23
            eval_text = "좋음"
    elif avg_angle >= 150:
        score = 20
        eval_text = "보통"
    elif avg_angle >= 140:
        score = 15
        eval_text = "약간 구부러짐"
    elif avg_angle >= 130:
        score = 10
        eval_text = "많이 구부러짐 (팔을 더 펼쳐주세요)"
    else:
        score = 5
        eval_text = "심하게 구부러짐 (팔을 훨씬 더 펼쳐주세요)"
    
    return {
        "score": score,
        "eval": eval_text,
        "detail": f"평균 팔 각도: {avg_angle:.1f}°",
        "average_angle": round(avg_angle, 1),
        "left_angle": round(left_arm_angle, 1) if left_arm_angle else None,
        "right_angle": round(right_arm_angle, 1) if right_arm_angle else None
    }


def evaluate_shoulder_level(left_shoulder, right_shoulder, torso_height=None):
    """
    C. 어깨 수평 (Shoulder Level) - 15점
    좌우 어깨 높이 차이 계산
    차이가 작을수록 좋음, 필요시 정규화
    """
    if left_shoulder is None or right_shoulder is None:
        return {"score": 0, "eval": "계산불가", "detail": "어깨 감지 불가", "difference_px": 0}
    
    shoulder_diff = abs(left_shoulder[1] - right_shoulder[1])
    
    # 정규화: torso_height가 제공되면 비율로 변환
    # 예: 어깨너비 기준 (대략 left_shoulder[0] - right_shoulder[0])
    shoulder_width = abs(left_shoulder[0] - right_shoulder[0])
    
    if shoulder_width > 0:
        normalized_diff = (shoulder_diff / shoulder_width) * 100  # 비율 %
    else:
        normalized_diff = shoulder_diff
    
    # 판정 기준
    if shoulder_diff < 5:
        score = 15
        eval_text = "완벽하게 수평"
    elif shoulder_diff < 10:
        score = 14
        eval_text = "거의 수평"
    elif shoulder_diff < 15:
        score = 12
        eval_text = "약간 기울어짐"
    elif shoulder_diff < 20:
        score = 10
        eval_text = "기울어짐"
    elif shoulder_diff < 30:
        score = 7
        eval_text = "크게 기울어짐 (어깨를 수평으로 맞춰주세요)"
    else:
        score = 4
        eval_text = "매우 크게 기울어짐"
    
    return {
        "score": score,
        "eval": eval_text,
        "detail": f"높이 차이: {shoulder_diff:.1f}px (비율: {normalized_diff:.1f}%)",
        "difference_px": round(shoulder_diff, 1),
        "normalized_percent": round(normalized_diff, 1)
    }


def evaluate_head_position(nose, shoulder_center):
    """
    D. 머리 위치 (Head Position) - 15점
    코가 어깨 중심 대비 위치 계산
    너무 빡빡하지 않은 범위 사용
    """
    if nose is None or shoulder_center is None:
        return {"score": 0, "eval": "계산불가", "detail": "코 또는 어깨 감지 불가"}
    
    # 상대 위치
    vertical_offset = nose[1] - shoulder_center[1]  # 음수 = 위쪽
    horizontal_offset = nose[0] - shoulder_center[0]  # 양수 = 오른쪽
    
    # 거리
    offset_distance = math.sqrt(vertical_offset**2 + horizontal_offset**2)
    
    # 판정 (너무 빡빡하지 않게)
    if offset_distance < 80:
        score = 15
        eval_text = "좋은 머리 위치"
    elif offset_distance < 120:
        score = 13
        eval_text = "괜찮은 머리 위치"
    elif offset_distance < 160:
        score = 10
        eval_text = "약간 벗어난 머리 위치 (중심으로 조정해주세요)"
    elif offset_distance < 200:
        score = 7
        eval_text = "크게 벗어난 머리 위치"
    else:
        score = 4
        eval_text = "매우 벗어난 머리 위치"
    
    return {
        "score": score,
        "eval": eval_text,
        "detail": f"오프셋: 세로 {vertical_offset:.1f}px, 가로 {horizontal_offset:.1f}px",
        "vertical_offset": round(vertical_offset, 1),
        "horizontal_offset": round(horizontal_offset, 1),
        "distance": round(offset_distance, 1)
    }


def calculate_linearity_score(shoulder, elbow, wrist):
    """
    E. 팔/손 정렬 (Arm/Wrist Alignment) - 25점
    어깨-팔꿈치-손목이 얼마나 일직선에 가까운지 계산
    직선성이 높을수록 높은 점수
    """
    if shoulder is None or elbow is None or wrist is None:
        return {"score": 0, "eval": "계산불가", "detail": "필요한 포인트 감지 불가", "linearity": 0}
    
    # 벡터: shoulder -> elbow, shoulder -> wrist
    v1 = np.array(elbow) - np.array(shoulder)
    v2 = np.array(wrist) - np.array(shoulder)
    
    # 직선성 계산: cos(각도)를 이용
    # 완벽한 직선 = 1, 수직 = 0
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    
    if norm_v1 == 0 or norm_v2 == 0:
        linearity = 0
    else:
        cos_angle = np.dot(v1, v2) / (norm_v1 * norm_v2)
        linearity = max(0, cos_angle)  # 음수 방지
    
    # 0 ~ 1 범위를 0 ~ 100 범위로 변환
    linearity_percent = linearity * 100
    
    # 판정 기준
    if linearity >= 0.95:
        score = 25
        eval_text = "완벽하게 일직선"
    elif linearity >= 0.90:
        score = 24
        eval_text = "거의 일직선"
    elif linearity >= 0.85:
        score = 22
        eval_text = "좋은 정렬"
    elif linearity >= 0.80:
        score = 20
        eval_text = "양호한 정렬"
    elif linearity >= 0.70:
        score = 16
        eval_text = "약간 기울어짐"
    elif linearity >= 0.60:
        score = 12
        eval_text = "상당히 기울어짐 (팔을 더 펴고 직선에 맞춰주세요)"
    else:
        score = 8
        eval_text = "심하게 굽어짐"
    
    return {
        "score": score,
        "eval": eval_text,
        "detail": f"직선성: {linearity_percent:.1f}%",
        "linearity": round(linearity, 3),
        "linearity_percent": round(linearity_percent, 1)
    }

def detect_body_orientation(keypoints, visibility_scores):
    """
    사람의 몸 방향 감지 (정면, 좌측, 우측)
    
    반환값:
    - "front": 정면을 보고 있음
    - "left": 왼쪽을 보고 있음 (우측 어깨가 더 앞쪽)
    - "right": 오른쪽을 보고 있음 (좌측 어깨가 더 앞쪽)
    """
    # 랜드마크 인덱스
    nose = 0
    left_eye = 2
    right_eye = 5
    left_shoulder = 11
    right_shoulder = 12
    
    # 가시성 확인
    if (visibility_scores[left_shoulder] < 0.3 or 
        visibility_scores[right_shoulder] < 0.3):
        return "front"  # 기본값
    
    # 어깨 너비 계산
    shoulder_x_diff = keypoints[right_shoulder][0] - keypoints[left_shoulder][0]
    
    # 어깨 깊이 (z축 느낌, y좌표로 표현)
    shoulder_y_diff = keypoints[right_shoulder][1] - keypoints[left_shoulder][1]
    
    # 코와 어깨 중심의 관계
    if visibility_scores[nose] > 0.3:
        shoulder_center_x = (keypoints[left_shoulder][0] + keypoints[right_shoulder][0]) / 2
        nose_x = keypoints[nose][0]
        
        nose_to_shoulder = nose_x - shoulder_center_x
        
        # 오른쪽을 보고 있음: 코가 어깨 중심보다 오른쪽
        if nose_to_shoulder > 30:
            return "right"
        # 왼쪽을 보고 있음: 코가 어깨 중심보다 왼쪽
        elif nose_to_shoulder < -30:
            return "left"
    
    return "front"


def analyze_pose(image_path):
    """
    사진에서 포즈 분석 (새로운 사격 자세 평가 기준)
    
    평가 항목 (총 100점):
    A. 상체 기울기 (Body Lean): 20점
    B. 팔 각도 (Arm Extension): 25점
    C. 어깨 수평 (Shoulder Level): 15점
    D. 머리 위치 (Head Position): 15점
    E. 팔/손 정렬 (Arm/Wrist Alignment): 25점
    """

    print(f"\n{'='*60}")
    print(f"[포즈 분석 시작 - 새로운 사격 자세 평가 기준]")
    print(f"{'='*60}")

    try:
        # 경로 정규화
        image_path = str(image_path)
        image_path = os.path.normpath(image_path)

        # 파일 존재 여부 확인
        if not os.path.exists(image_path):
            return {"error": "파일을 찾을 수 없습니다"}

        # 이미지 읽기
        image = cv2.imread(image_path)
        if image is None:
            return {"error": "이미지를 읽을 수 없습니다"}

        print(f"✓ 이미지 읽음: {image.shape}")

        # 이미지 전처리
        try:
            enhanced_image = enhance_image(image)
            print(f"✓ 이미지 전처리 완료")
        except Exception as e:
            print(f"WARNING: 전처리 실패, 원본 사용: {e}")
            enhanced_image = image

        # MediaPipe 포즈 모델 로드
        pose_model = get_pose_model()

        # MediaPipe 포즈 감지
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(enhanced_image, cv2.COLOR_BGR2RGB))
        results = pose_model.detect(mp_image)

        if not results.pose_landmarks:
            print("✗ 포즈 감지 실패")
            return {
                "status": "failed",
                "posture_grade": "N/A",
                "overall_score": 0,
                "error": "사람의 자세를 감지할 수 없습니다"
            }

        # MediaPipe 랜드마크 추출
        landmarks = results.pose_landmarks[0]
        h, w, _ = enhanced_image.shape

        # 랜드마크를 픽셀 좌표로 변환
        keypoints = []
        visibility_scores = []

        for landmark in landmarks:
            x = int(landmark.x * w)
            y = int(landmark.y * h)
            visibility = landmark.visibility
            keypoints.append([x, y])
            visibility_scores.append(visibility)

        keypoints = np.array(keypoints)
        visibility_scores = np.array(visibility_scores)

        # 신체 방향 감지
        body_orientation = detect_body_orientation(keypoints, visibility_scores)
        print(f"[신체 방향] Body Orientation: {body_orientation}")

        # 주요 포인트 추출 (가시성 임계값: 0.3)
        def get_point(idx):
            if visibility_scores[idx] > 0.3:
                return keypoints[idx]
            return None

        nose = get_point(0)
        left_shoulder = get_point(11)
        right_shoulder = get_point(12)
        left_elbow = get_point(13)
        right_elbow = get_point(14)
        left_wrist = get_point(15)
        right_wrist = get_point(16)
        left_hip = get_point(23)
        right_hip = get_point(24)

        print(f"\n[주요 포인트 감지]")
        print(f"✓ 코: {nose is not None}")
        print(f"✓ 어깨: {left_shoulder is not None}, {right_shoulder is not None}")
        print(f"✓ 팔꿈치: {left_elbow is not None}, {right_elbow is not None}")
        print(f"✓ 손목: {left_wrist is not None}, {right_wrist is not None}")
        print(f"✓ 엉덩이: {left_hip is not None}, {right_hip is not None}")

        # 결과 딕셔너리
        results_dict = {
            "status": "success",
            "scores": {},
            "body_orientation": body_orientation
        }

        # ============ A. 상체 기울기 (Body Lean) - 20점 ============
        print(f"\n[A. 상체 기울기 분석]")
        shoulder_center = None
        hip_center = None
        
        if left_shoulder is not None and right_shoulder is not None:
            shoulder_center = ((left_shoulder[0] + right_shoulder[0]) / 2,
                              (left_shoulder[1] + right_shoulder[1]) / 2)
        
        if left_hip is not None and right_hip is not None:
            hip_center = ((left_hip[0] + right_hip[0]) / 2,
                         (left_hip[1] + right_hip[1]) / 2)
        
        body_lean_result = evaluate_body_lean(shoulder_center, hip_center)
        results_dict["scores"]["body_lean"] = body_lean_result
        print(f"평가: {body_lean_result['eval']}")
        print(f"상세: {body_lean_result['detail']}")
        print(f"점수: {body_lean_result['score']}/20")

        # ============ B. 팔 각도 (Arm Extension) - 25점 ============
        print(f"\n[B. 팔 각도 분석]")
        left_arm_angle = None
        right_arm_angle = None

        if left_shoulder is not None and left_elbow is not None and left_wrist is not None:
            left_arm_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
            print(f"✓ 왼팔 각도: {left_arm_angle:.1f}°")

        if right_shoulder is not None and right_elbow is not None and right_wrist is not None:
            right_arm_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)
            print(f"✓ 오른팔 각도: {right_arm_angle:.1f}°")

        arm_extension_result = evaluate_arm_extension(left_arm_angle, right_arm_angle)
        results_dict["scores"]["arm_extension"] = arm_extension_result
        print(f"평가: {arm_extension_result['eval']}")
        print(f"상세: {arm_extension_result['detail']}")
        print(f"점수: {arm_extension_result['score']}/25")

        # ============ C. 어깨 수평 (Shoulder Level) - 15점 ============
        print(f"\n[C. 어깨 수평 분석]")
        shoulder_level_result = evaluate_shoulder_level(left_shoulder, right_shoulder)
        results_dict["scores"]["shoulder_level"] = shoulder_level_result
        print(f"평가: {shoulder_level_result['eval']}")
        print(f"상세: {shoulder_level_result['detail']}")
        print(f"점수: {shoulder_level_result['score']}/15")

        # ============ D. 머리 위치 (Head Position) - 15점 ============
        print(f"\n[D. 머리 위치 분석]")
        head_position_result = evaluate_head_position(nose, shoulder_center)
        results_dict["scores"]["head_position"] = head_position_result
        print(f"평가: {head_position_result['eval']}")
        print(f"상세: {head_position_result['detail']}")
        print(f"점수: {head_position_result['score']}/15")

        # ============ E. 팔/손 정렬 (Arm/Wrist Alignment) - 25점 ============
        print(f"\n[E. 팔/손 정렬 분석]")
        
        # 왼팔과 오른팔의 평균 점수
        left_arm_linearity = calculate_linearity_score(left_shoulder, left_elbow, left_wrist)
        right_arm_linearity = calculate_linearity_score(right_shoulder, right_elbow, right_wrist)
        
        # 양쪽 모두 감지되면 평균, 한쪽만 감지되면 그쪽만 사용
        linearity_scores = []
        if left_arm_linearity["score"] > 0:
            linearity_scores.append(left_arm_linearity["score"])
            print(f"왼팔 정렬: {left_arm_linearity['eval']} ({left_arm_linearity['detail']})")
        
        if right_arm_linearity["score"] > 0:
            linearity_scores.append(right_arm_linearity["score"])
            print(f"오른팔 정렬: {right_arm_linearity['eval']} ({right_arm_linearity['detail']})")
        
        if linearity_scores:
            avg_linearity_score = sum(linearity_scores) / len(linearity_scores)
            arm_wrist_result = {
                "score": int(avg_linearity_score),
                "eval": "팔과 손목이 적절히 정렬됨",
                "detail": f"양팔 정렬도 평균: {avg_linearity_score:.1f}/25",
                "left_arm": left_arm_linearity,
                "right_arm": right_arm_linearity
            }
        else:
            arm_wrist_result = {
                "score": 0,
                "eval": "계산불가",
                "detail": "필요한 포인트 감지 불가"
            }
        
        results_dict["scores"]["arm_wrist_alignment"] = arm_wrist_result
        print(f"점수: {arm_wrist_result['score']}/25")

        # ============ 종합 평가 ============
        print(f"\n{'='*60}")
        print(f"[종합 평가]")
        print(f"{'='*60}")
        
        total_score = (
            body_lean_result["score"] +
            arm_extension_result["score"] +
            shoulder_level_result["score"] +
            head_position_result["score"] +
            arm_wrist_result["score"]
        )

        # 등급 결정
        if total_score >= 90:
            grade = "A"
            grade_text = "우수한 자세"
        elif total_score >= 80:
            grade = "B"
            grade_text = "좋은 자세"
        elif total_score >= 70:
            grade = "C"
            grade_text = "보통 자세"
        elif total_score >= 60:
            grade = "D"
            grade_text = "개선 필요"
        else:
            grade = "F"
            grade_text = "심각한 문제"

        results_dict["overall_score"] = total_score
        results_dict["posture_grade"] = grade
        results_dict["grade_text"] = grade_text
        
        print(f"총점: {total_score}/100")
        print(f"등급: {grade} ({grade_text})")

        # ============ 피드백 생성 ============
        print(f"\n[피드백]")
        feedback_list = generate_shooting_feedback(results_dict)
        results_dict["feedback"] = feedback_list
        
        for feedback in feedback_list:
            print(f"• {feedback}")

        # ============ 시각화 ============
        try:
            print(f"[시각화] 시각화 시작...")
            vis_image = draw_pose_landmarks(image, landmarks)
            
            if vis_image is None:
                print(f"[시각화] ERROR: draw_pose_landmarks가 None 반환")
                results_dict["visualization_image"] = None
            else:
                base_name = os.path.splitext(os.path.basename(image_path))[0]
                vis_filename = f"{base_name}_landmarks.png"  # PNG로 변경 (품질 손실 없음)
                vis_path = os.path.join(os.path.dirname(image_path), vis_filename)
                
                print(f"[시각화] 저장 경로: {vis_path}")
                print(f"[시각화] 이미지 정보: shape={vis_image.shape}, dtype={vis_image.dtype}")
                
                success = cv2.imwrite(vis_path, vis_image)
                
                if success:
                    file_size = os.path.getsize(vis_path)
                    results_dict["visualization_image"] = vis_filename
                    print(f"✓ 시각화 이미지 저장 완료: {vis_filename} ({file_size} bytes)")
                else:
                    print(f"[시각화] ERROR: cv2.imwrite 실패")
                    results_dict["visualization_image"] = None
        except Exception as e:
            print(f"[시각화] ERROR: {e}")
            import traceback
            traceback.print_exc()
            results_dict["visualization_image"] = None

        print(f"{'='*60}\n")
        return results_dict

    except Exception as e:
        print(f"✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return {"error": str(e), "status": "error"}