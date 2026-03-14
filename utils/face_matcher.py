"""얼굴 임베딩 추출 및 닮은꼴 매칭"""
import os
import pickle
import numpy as np
import cv2
from deepface import DeepFace

# 프로젝트 경로
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CELEB_DIR = os.path.join(BASE_DIR, "celebrities")
EMBED_DIR = os.path.join(BASE_DIR, "embeddings")
CACHE_FILE = os.path.join(EMBED_DIR, "celeb_embeddings.pkl")

# 사용 모델 (Facenet - PyTorch 기반, Streamlit Cloud 메모리 절약)
MODEL_NAME = "Facenet"
DETECTOR_BACKEND = "opencv"


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """코사인 유사도 계산 (numpy 직접 구현)"""
    dot = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(dot / (norm_a * norm_b))


def _read_image(image_path: str) -> np.ndarray:
    """한글 경로 지원하는 이미지 읽기 (cv2.imread는 한글 경로 미지원)"""
    with open(image_path, "rb") as f:
        data = np.frombuffer(f.read(), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    # deepface는 RGB 기대
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _extract_embedding(image_path: str) -> np.ndarray | None:
    """이미지에서 얼굴 임베딩 추출"""
    img_array = _read_image(image_path)
    try:
        result = DeepFace.represent(
            img_path=img_array,
            model_name=MODEL_NAME,
            detector_backend=DETECTOR_BACKEND,
            enforce_detection=True,
        )
        return np.array(result[0]["embedding"])
    except Exception:
        try:
            result = DeepFace.represent(
                img_path=img_array,
                model_name=MODEL_NAME,
                detector_backend=DETECTOR_BACKEND,
                enforce_detection=False,
            )
            return np.array(result[0]["embedding"])
        except Exception:
            return None


def build_celeb_embeddings() -> dict:
    """celebrities/ 폴더의 모든 사진에서 임베딩 추출 후 캐시"""
    # 캐시 파일이 있으면 로드
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "rb") as f:
            cached = pickle.load(f)
        # 사진 변경 확인 (파일 수 비교)
        current_files = _get_celeb_files()
        if len(cached) == len(current_files):
            return cached

    # 새로 생성
    embeddings = {}
    celeb_files = _get_celeb_files()

    for name, path in celeb_files.items():
        print(f"  임베딩 추출 중: {name}")
        emb = _extract_embedding(path)
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


def find_lookalikes(user_image, celeb_embeddings: dict, top_n: int = 3) -> list[dict]:
    """사용자 이미지와 연예인 임베딩 비교 후 TOP N 반환

    Args:
        user_image: numpy 배열 (RGB) 또는 이미지 파일 경로
        celeb_embeddings: build_celeb_embeddings() 결과
        top_n: 반환할 상위 결과 수

    Returns:
        [{"name": str, "similarity": float, "image_path": str}, ...]
    """
    # 사용자 얼굴 임베딩 추출
    try:
        result = DeepFace.represent(
            img_path=user_image,
            model_name=MODEL_NAME,
            detector_backend=DETECTOR_BACKEND,
            enforce_detection=True,
        )
        user_embedding = np.array(result[0]["embedding"])
    except Exception:
        try:
            result = DeepFace.represent(
                img_path=user_image,
                model_name=MODEL_NAME,
                detector_backend=DETECTOR_BACKEND,
                enforce_detection=False,
            )
            user_embedding = np.array(result[0]["embedding"])
        except Exception:
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
