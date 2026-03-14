# 🪞 닮은꼴 연예인 찾기

> AI가 찾아주는 나의 연예인 쌍둥이!

셀카를 올리면 가장 닮은 한국 연예인 TOP 3를 찾아주는 웹앱입니다.

## 기술 스택

| 구분 | 기술 |
|------|------|
| **프레임워크** | Streamlit |
| **얼굴 분석** | DeepFace (VGG-Face 모델) |
| **이미지 처리** | OpenCV, Pillow |
| **언어** | Python 3.9+ |
| **배포** | Streamlit Cloud |

## 주요 기능

- 📸 셀카 업로드 → AI 얼굴 분석
- 🏆 닮은꼴 연예인 TOP 3 (유사도 % 표시)
- ⭐ 한국 연예인 20명 등록 (남녀 각 10명)
- 🎨 다크 테마 감성 UI

## 로컬 실행 방법

```bash
# 1. 패키지 설치
pip install -r requirements.txt

# 2. 앱 실행
streamlit run app.py
```

브라우저에서 `http://localhost:8501` 접속

## 연예인 사진 크롤링 (선택)

celebrities/ 폴더의 사진을 새로 수집하려면:

```bash
# 개발용 패키지 설치
pip install -r requirements-dev.txt

# 크롤링 실행
python scripts/crawl_celebrities.py
```

## Streamlit Cloud 배포 방법

1. GitHub에 이 레포를 push
2. [share.streamlit.io](https://share.streamlit.io) 접속
3. **New app** → 레포 선택 → `app.py` 지정 → **Deploy**

## 등록된 연예인

**여성**: 아이유, 수지, 제니, 카리나, 한소희, 김태리, 전지현, 박은빈, 김유정, 장원영

**남성**: 차은우, 뷔, 공유, 현빈, 박서준, 송중기, 이도현, 박보검, 정해인, 안효섭

---

Made with ❤️ by 동원
