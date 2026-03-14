"""이미지 전처리 유틸리티"""
from PIL import Image, ExifTags
import io
import numpy as np


def fix_exif_rotation(image: Image.Image) -> Image.Image:
    """EXIF 회전 정보 보정 (모바일 셀카 대응)"""
    try:
        exif = image._getexif()
        if exif is None:
            return image

        # EXIF orientation 태그 찾기
        orientation_key = None
        for key, val in ExifTags.TAGS.items():
            if val == "Orientation":
                orientation_key = key
                break

        if orientation_key is None or orientation_key not in exif:
            return image

        orientation = exif[orientation_key]
        # 회전 보정 매핑
        if orientation == 3:
            image = image.rotate(180, expand=True)
        elif orientation == 6:
            image = image.rotate(270, expand=True)
        elif orientation == 8:
            image = image.rotate(90, expand=True)
    except (AttributeError, KeyError):
        pass
    return image


def resize_image(image: Image.Image, max_size: int = 800) -> Image.Image:
    """이미지를 최대 크기 이내로 리사이즈"""
    w, h = image.size
    if w <= max_size and h <= max_size:
        return image
    ratio = min(max_size / w, max_size / h)
    new_size = (int(w * ratio), int(h * ratio))
    return image.resize(new_size, Image.LANCZOS)


def process_uploaded_image(uploaded_file) -> np.ndarray:
    """업로드 이미지를 전처리 후 numpy 배열로 변환"""
    image = Image.open(uploaded_file)

    # RGBA → RGB 변환
    if image.mode == "RGBA":
        background = Image.new("RGB", image.size, (255, 255, 255))
        background.paste(image, mask=image.split()[3])
        image = background
    elif image.mode != "RGB":
        image = image.convert("RGB")

    # EXIF 회전 보정
    image = fix_exif_rotation(image)

    # 리사이즈
    image = resize_image(image)

    return np.array(image)


def image_to_bytes(image: Image.Image, fmt: str = "JPEG") -> bytes:
    """PIL Image를 바이트로 변환"""
    buf = io.BytesIO()
    image.save(buf, format=fmt)
    return buf.getvalue()
