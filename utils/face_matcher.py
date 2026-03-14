"""얼굴 임베딩 추출 및 닮은꼴 매칭 (OpenCV SFace 기반, tensorflow 불필요)"""
import os
import pickle
import requests
import numpy as np
import cv2

# 프로젝트 경로
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CELEB_DIR = os.path.join(BASE_DIR, "celebrities")
EMBED_DIR = os.path.join(BASE_DIR, "embeddings")
CACHE_FILE = os.path.join(EMBED_DIR, "celeb_embeddings.pkl")
MODEL_DIR = os.path.join(BASE_DIR, "models")

# OpenCV Zoo 모델 URL
YUNET_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
SFACE_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx"

YUNET_PATH = os.path.join(MODEL_DIR, "face_detection_yunet_2023mar.onnx")
SFACE_PATH = os.path.join(MODEL_DIR, "face_recognition_sface_2021dec.onnx")


def _download_model(url: str, path: str):
    """모델 파일 다운로드"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if os.path.exists(path):
        return
    print(f"  모델 다운로드 중: {os.path.basename(path)}")
    resp = requests.get(url, allow_redirects=True, timeout=120)
    resp.raise_for_status()
    with open(path, "wb") as f:
        f.write(resp.content)
    print(f"  다운로드 완료: {os.path.basename(path)}")


def _ensure_models():
    """모델 파일 존재 확인 및 다운로드"""
    _download_model(YUNET_URL, YUNET_PATH)
    _download_model(SFACE_URL, SFACE_PATH)


def _create_detector(input_size=(300, 300)):
    """YuNet 얼굴 감지기 생성"""
    detector = cv2.FaceDetectorYN.create(
        YUNET_PATH,
        "",
        input_size,
        score_threshold=0.5,
        nms_threshold=0.3,
        top_k=5,
    )
    return detector


def _create_recognizer():
    """SFace 얼굴 인식기 생성"""
    recognizer = cv2.FaceRecognizerSF.create(SFACE_PATH, "")
    return recognizer


def _read_image(image_path: str) -> np.ndarray:
    """한글 경로 지원하는 이미지 읽기"""
    with open(image_path, "rb") as f:
        data = np.frombuffer(f.read(), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    return img


def _detect_and_align(image: np.ndarray, detector, recognizer) -> np.ndarray | None:
    """이미지에서 얼굴 감지 후 정렬된 얼굴 반환"""
    h, w = image.shape[:2]
    detector.setInputSize((w, h))
    _, faces = detector.detect(image)

    if faces is None or len(faces) == 0:
        return None

    # 가장 큰 얼굴 선택
    biggest_idx = 0
    biggest_area = 0
    for i, face in enumerate(faces):
        area = face[2] * face[3]  # width * height
        if area > biggest_area:
            biggest_area = area
            biggest_idx = i

    aligned = recognizer.alignCrop(image, faces[biggest_idx])
    return aligned


def _extract_embedding(image: np.ndarray, detector, recognizer) -> np.ndarray | None:
    """이미지에서 얼굴 임베딩 추출"""
    aligned = _detect_and_align(image, detector, recognizer)
    if aligned is None:
        return None
    feature = recognizer.feature(aligned)
    return feature.flatten()


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """코사인 유사도 계산"""
    dot = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(dot / (norm_a * norm_b))


def _get_celeb_files() -> dict:
    """celebrities/ 폴더에서 이미지 파일 목록 반환"""
    files = {}
    if not os.path.exists(CELEB_DIR):
        return files
    for fname in os.listdir(CELEB_DIR):
        if fname.lower().endswith((".jpg", ".jpeg", ".png")):
            name = os.path.splitext(fname)[0]
            files[name] = os.path.join(CELEB_DIR, fname)
    return files


def build_celeb_embeddings() -> dict:
    """celebrities/ 폴더의 모든 사진에서 임베딩 추출 후 캐시"""
    _ensure_models()

    # 캐시 파일이 있으면 로드
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "rb") as f:
            cached = pickle.load(f)
        current_files = _get_celeb_files()
        if len(cached) == len(current_files):
            return cached

    # 새로 생성
    detector = _create_detector()
    recognizer = _create_recognizer()
    embeddings = {}
    celeb_files = _get_celeb_files()

    for name, path in celeb_files.items():
        print(f"  임베딩 추출 중: {name}")
        image = _read_image(path)
        if image is None:
            print(f"  ⚠️ 이미지 읽기 실패: {name}")
            continue
        emb = _extract_embedding(image, detector, recognizer)
        if emb is not None:
            embeddings[name] = {
                "embedding": emb,
                "image_path": path,
            }
        else:
            print(f"  ⚠️ 얼굴 감지 실패: {name}")

    # 캐시 저장
    os.makedirs(EMBED_DIR, exist_ok=True)
    with open(CACHE_FILE, "wb") as f:
        pickle.dump(embeddings, f)

    return embeddings


def find_lookalikes(user_image_path: str, celeb_embeddings: dict, top_n: int = 3) -> list[dict]:
    """사용자 이미지와 연예인 임베딩 비교 후 TOP N 반환"""
    _ensure_models()
    detector = _create_detector()
    recognizer = _create_recognizer()

    # 사용자 이미지 읽기
    image = cv2.imread(user_image_path)
    if image is None:
        # 한글 경로 대비
        image = _read_image(user_image_path)
    if image is None:
        return []

    # 사용자 얼굴 임베딩 추출
    user_embedding = _extract_embedding(image, detector, recognizer)
    if user_embedding is None:
        return []

    # 모든 연예인과 유사도 비교
    similarities = []
    for name, data in celeb_embeddings.items():
        sim = _cosine_similarity(user_embedding, data["embedding"])
        # 유사도를 0~100% 스케일로 변환
        similarity_pct = round(max(0, sim) * 100, 1)
        similarities.append({
            "name": name,
            "similarity": similarity_pct,
            "image_path": data["image_path"],
        })

    # 유사도 높은 순 정렬
    similarities.sort(key=lambda x: x["similarity"], reverse=True)
    return similarities[:top_n]
