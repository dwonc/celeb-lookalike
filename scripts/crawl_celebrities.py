"""연예인 프로필 사진 자동 크롤링 스크립트"""
import os
import sys
import shutil
import tempfile
from PIL import Image

# 프로젝트 루트 경로
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CELEB_DIR = os.path.join(BASE_DIR, "celebrities")

# 연예인 리스트
celebrities = {
    "아이유": "아이유 가수 프로필 사진 정면",
    "수지": "수지 배우 프로필 사진 정면",
    "제니": "제니 블랙핑크 프로필 사진 정면",
    "카리나": "카리나 에스파 프로필 사진 정면",
    "한소희": "한소희 배우 프로필 사진 정면",
    "김태리": "김태리 배우 프로필 사진 정면",
    "전지현": "전지현 배우 프로필 사진 정면",
    "박은빈": "박은빈 배우 프로필 사진 정면",
    "김유정": "김유정 배우 프로필 사진 정면",
    "장원영": "장원영 아이브 프로필 사진 정면",
    "차은우": "차은우 배우 프로필 사진 정면",
    "뷔": "뷔 방탄소년단 프로필 사진 정면",
    "공유": "공유 배우 프로필 사진 정면",
    "현빈": "현빈 배우 프로필 사진 정면",
    "박서준": "박서준 배우 프로필 사진 정면",
    "송중기": "송중기 배우 프로필 사진 정면",
    "이도현": "이도현 배우 프로필 사진 정면",
    "박보검": "박보검 배우 프로필 사진 정면",
    "정해인": "정해인 배우 프로필 사진 정면",
    "안효섭": "안효섭 배우 프로필 사진 정면",
}


def select_best_image(temp_dir: str) -> str | None:
    """다운로드된 이미지 중 해상도가 가장 큰 것 선택"""
    best_path = None
    best_pixels = 0

    for fname in os.listdir(temp_dir):
        fpath = os.path.join(temp_dir, fname)
        try:
            with Image.open(fpath) as img:
                pixels = img.size[0] * img.size[1]
                if pixels > best_pixels:
                    best_pixels = pixels
                    best_path = fpath
        except Exception:
            continue

    return best_path


def resize_and_save(src_path: str, dst_path: str, max_size: int = 500):
    """이미지를 최대 500x500으로 리사이즈 후 JPG로 저장"""
    with Image.open(src_path) as img:
        if img.mode != "RGB":
            img = img.convert("RGB")
        w, h = img.size
        if w > max_size or h > max_size:
            ratio = min(max_size / w, max_size / h)
            img = img.resize((int(w * ratio), int(h * ratio)), Image.LANCZOS)
        img.save(dst_path, "JPEG", quality=90)


def crawl_with_icrawler(query: str, temp_dir: str, max_num: int = 3) -> bool:
    """icrawler의 BingImageCrawler로 크롤링"""
    try:
        from icrawler.builtin import BingImageCrawler
        crawler = BingImageCrawler(
            storage={"root_dir": temp_dir},
            log_level=40,  # ERROR만 표시
        )
        crawler.crawl(keyword=query, max_num=max_num)
        return len(os.listdir(temp_dir)) > 0
    except Exception as e:
        print(f"    icrawler 실패: {e}")
        return False


def crawl_with_bing_downloader(query: str, temp_dir: str, max_num: int = 3) -> bool:
    """bing-image-downloader로 크롤링"""
    try:
        from bing_image_downloader import downloader
        downloader.download(
            query, limit=max_num,
            output_dir=temp_dir,
            adult_filter_off=False,
            force_replace=True,
            timeout=10,
        )
        # bing-image-downloader는 하위 폴더에 저장
        sub_dir = os.path.join(temp_dir, query)
        if os.path.exists(sub_dir):
            for f in os.listdir(sub_dir):
                shutil.move(os.path.join(sub_dir, f), os.path.join(temp_dir, f))
            shutil.rmtree(sub_dir)
        return len([f for f in os.listdir(temp_dir) if not os.path.isdir(os.path.join(temp_dir, f))]) > 0
    except Exception as e:
        print(f"    bing-image-downloader 실패: {e}")
        return False


def crawl_with_duckduckgo(query: str, temp_dir: str, max_num: int = 3) -> bool:
    """duckduckgo_search로 크롤링"""
    try:
        from duckduckgo_search import DDGS
        import requests

        with DDGS() as ddgs:
            results = list(ddgs.images(query, max_results=max_num))

        count = 0
        for i, result in enumerate(results):
            try:
                resp = requests.get(result["image"], timeout=10,
                                    headers={"User-Agent": "Mozilla/5.0"})
                if resp.status_code == 200 and len(resp.content) > 1000:
                    ext = ".jpg"
                    filepath = os.path.join(temp_dir, f"{i}{ext}")
                    with open(filepath, "wb") as f:
                        f.write(resp.content)
                    # 유효한 이미지인지 확인
                    Image.open(filepath).verify()
                    count += 1
            except Exception:
                continue
        return count > 0
    except Exception as e:
        print(f"    duckduckgo_search 실패: {e}")
        return False


def crawl_celebrity(name: str, query: str) -> bool:
    """연예인 1명 크롤링 (폴백 포함)"""
    dst_path = os.path.join(CELEB_DIR, f"{name}.jpg")

    # 이미 존재하면 스킵
    if os.path.exists(dst_path):
        print(f"  ✅ {name} - 이미 존재, 스킵")
        return True

    temp_dir = tempfile.mkdtemp()
    try:
        # 1순위: icrawler
        success = crawl_with_icrawler(query, temp_dir)

        # 2순위: bing-image-downloader
        if not success:
            print(f"    → bing-image-downloader 시도...")
            success = crawl_with_bing_downloader(query, temp_dir)

        # 3순위: duckduckgo_search
        if not success:
            print(f"    → duckduckgo_search 시도...")
            success = crawl_with_duckduckgo(query, temp_dir)

        if not success:
            print(f"  ❌ {name} - 모든 크롤러 실패")
            return False

        # 가장 좋은 이미지 선택
        best = select_best_image(temp_dir)
        if best is None:
            print(f"  ❌ {name} - 유효한 이미지 없음")
            return False

        # 리사이즈 후 저장
        resize_and_save(best, dst_path)
        print(f"  ✅ {name} - 저장 완료")
        return True

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def main():
    os.makedirs(CELEB_DIR, exist_ok=True)

    print("🎬 연예인 프로필 사진 크롤링 시작!")
    print(f"   대상: {len(celebrities)}명")
    print(f"   저장 위치: {CELEB_DIR}")
    print()

    success_list = []
    fail_list = []

    for name, query in celebrities.items():
        print(f"📥 {name} 크롤링 중...")
        if crawl_celebrity(name, query):
            success_list.append(name)
        else:
            fail_list.append(name)

    print()
    print("=" * 50)
    print(f"✅ 성공: {len(success_list)}명")
    print(f"❌ 실패: {len(fail_list)}명")
    if fail_list:
        print(f"   실패 목록: {', '.join(fail_list)}")
    print(f"📁 저장된 사진: {len(os.listdir(CELEB_DIR))}장")


if __name__ == "__main__":
    main()
