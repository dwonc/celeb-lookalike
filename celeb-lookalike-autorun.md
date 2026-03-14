# 닮은꼴 연예인 찾기 - 전체 자동 빌드

너는 지금부터 이 프로젝트를 처음부터 배포까지 전부 자동으로 진행해야 해.
각 단계를 순서대로 실행하고, 에러가 나면 스스로 해결해. 질문하지 말고 최대한 알아서 판단해서 진행해.

## 프로젝트 정보
- 프로젝트명: celeb-lookalike
- 설명: 셀카를 올리면 가장 닮은 한국 연예인 TOP 3를 찾아주는 웹앱
- 기술스택: Python, Streamlit, deepface, OpenCV, Pillow, numpy
- 배포: Streamlit Cloud (GitHub 연동)
- 언어: 한국어 UI, 한국어 주석

---

## STEP 0: 프로젝트 구조 생성

아래 구조로 프로젝트를 생성해줘:

```
celeb-lookalike/
├── app.py
├── utils/
│   ├── __init__.py
│   ├── face_matcher.py
│   └── image_processor.py
├── scripts/
│   └── crawl_celebrities.py
├── celebrities/
├── embeddings/
├── requirements.txt
├── requirements-dev.txt
├── packages.txt
├── .streamlit/
│   └── config.toml
├── .gitignore
├── CLAUDE.md
└── README.md
```

requirements.txt (배포용):
```
streamlit>=1.30.0
deepface>=0.0.93
tf-keras
pillow>=10.0.0
numpy>=1.24.0
pandas>=2.0.0
opencv-python-headless>=4.8.0
```

requirements-dev.txt (개발용):
```
-r requirements.txt
icrawler>=0.6.7
```

packages.txt:
```
libgl1-mesa-glx
libglib2.0-0
```

.streamlit/config.toml:
```toml
[theme]
primaryColor = "#FF4B4B"
backgroundColor = "#0E1117"
secondaryBackgroundColor = "#262730"
textColor = "#FAFAFA"
font = "sans serif"

[server]
maxUploadSize = 10
```

.gitignore:
```
__pycache__/
*.pyc
.env
embeddings/*.pkl
*.egg-info/
dist/
build/
.DS_Store
```

CLAUDE.md:
```markdown
# 닮은꼴 연예인 찾기 프로젝트

## 규칙
- 한국어로 소통, 코드에 한국어 주석
- 에러 발생시 원인 분석 먼저 하고 수정

## 기술 스택
- Python 3.9+ / Streamlit / deepface / OpenCV / numpy

## 주의사항
- Streamlit Cloud 무료 메모리 1GB 제한
- 모델 로딩시 반드시 st.cache_resource 사용
- celebrities/ 폴더 사진은 git에 포함
- embeddings/ 캐시는 git에서 제외
```

---

## STEP 1: 연예인 사진 자동 크롤링

scripts/crawl_celebrities.py를 만들고 실행해줘.

기능:
- icrawler의 BingImageCrawler로 연예인 프로필 사진 자동 다운로드
- 연예인 1명당 3장 다운받고, Pillow로 해상도가 가장 큰 1장만 선별
- 최종 사진은 celebrities/ 폴더에 "이름.jpg" 형식으로 저장
- 500x500 이내로 리사이즈

연예인 리스트와 검색 쿼리:
```python
celebrities = {
    # "저장할_이름": "검색_쿼리"
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
```

pip install -r requirements-dev.txt 먼저 하고, 스크립트 실행해줘.

만약 icrawler가 실패하면 대체 방안:
1. bing-image-downloader 패키지 시도
2. duckduckgo_search 패키지로 DuckDuckGo 이미지 검색 시도
3. requests + BeautifulSoup으로 직접 크롤링

실행 후 celebrities/ 폴더에 사진이 몇 장 저장됐는지 알려줘.
크롤링 실패한 연예인이 있으면 목록으로 알려줘.

---

## STEP 2: 핵심 로직 구현

### utils/face_matcher.py

기능:
1. celebrities/ 폴더의 모든 연예인 사진에서 deepface로 얼굴 임베딩 추출
2. 임베딩을 embeddings/ 폴더에 pkl로 캐시 (매번 재계산 방지)
3. 사용자 셀카 업로드시 임베딩 추출 후 코사인 유사도로 TOP 3 매칭

상세:
- deepface 모델: "VGG-Face" (정확도 높음)
- 백엔드: opencv
- represent() 함수로 임베딩 추출
- 코사인 유사도는 numpy로 직접 구현
- 캐시 파일이 있으면 로드, 없으면 새로 생성
- 얼굴 미감지시 enforce_detection=False로 재시도, 그래도 안 되면 에러 메시지 반환

반환 형태:
```python
[
    {"name": "아이유", "similarity": 87.3, "image_path": "celebrities/아이유.jpg"},
    {"name": "수지", "similarity": 82.1, "image_path": "celebrities/수지.jpg"},
    {"name": "카리나", "similarity": 78.5, "image_path": "celebrities/카리나.jpg"}
]
```

### utils/image_processor.py

기능:
- 업로드 이미지 리사이즈 (최대 800x800)
- EXIF 회전 보정 (모바일 셀카 대응)
- 이미지 포맷 변환 (png→jpg 등)

---

## STEP 3: UI 구현

app.py를 재미있고 감성적인 UI로 구현해줘.

구성:
1. 헤더
   - 제목: "🪞 닮은꼴 연예인 찾기"
   - 부제: "AI가 찾아주는 나의 연예인 쌍둥이!"

2. 사이드바
   - 등록된 연예인 목록 (사진 썸네일 + 이름)
   - 총 등록 인원 수

3. 메인 영역
   - 셀카 업로드 (st.file_uploader, jpg/png/jpeg)
   - 업로드하면 내 사진 표시
   - "분석 시작! 🔍" 버튼
   - 분석 중 스피너 + 랜덤 로딩 메시지:
     "AI가 열심히 얼굴을 분석하고 있어요...",
     "연예인 데이터베이스를 뒤지는 중...",
     "당신의 숨겨진 매력을 찾는 중...",
     "혹시 연예인 아니세요...?"

4. 결과 영역
   - TOP 3를 3개 컬럼으로
   - 각 컬럼: 연예인 사진 + 이름 + 유사도(%) + 프로그레스 바
   - 1위: 🥇 + 크게 강조 + st.balloons()
   - "이 결과를 친구에게 공유하세요! 📱"

5. 푸터
   - "Made with ❤️ by 동원"
   - "※ 이 서비스는 재미용입니다. AI 분석 결과는 참고만 해주세요!"

스타일:
- st.markdown으로 커스텀 CSS (그라데이션 결과 카드, 1위 반짝이 애니메이션)
- 다크 테마 기본

성능:
- st.cache_resource로 임베딩 캐시
- 이미지 리사이즈로 업로드 최적화

---

## STEP 4: 크롤링 사진 검증 + 테스트

1. deepface로 celebrities/ 폴더의 각 사진에서 얼굴 감지 테스트
2. 감지 실패하는 사진은 다른 검색어로 재크롤링 시도
3. streamlit run app.py로 로컬 실행 테스트
4. 에러 있으면 전부 수정
5. 임베딩 캐시 정상 생성 확인

---

## STEP 5: Git + README

1. git init && git add . && git commit -m "feat: 닮은꼴 연예인 찾기 MVP"
2. README.md에 프로젝트 소개, 기술 스택, 실행 방법, 배포 방법, 크롤링 스크립트 사용법 작성

이 시점에서 나에게 알려줘:
- GitHub에 새 레포 만들고 push하라고 안내
- Streamlit Cloud 배포 방법 안내 (share.streamlit.io → New app → 레포 선택 → Deploy)

---

## 에러 대응 규칙

- icrawler 실패 → bing-image-downloader → duckduckgo_search → requests 순서로 시도
- deepface 얼굴 미감지 → enforce_detection=False로 재시도
- Streamlit Cloud 메모리 초과 → 모델을 "VGG-Face"에서 "Facenet"으로 변경
- import 에러 → requirements.txt 누락 확인 후 추가
- OpenCV 에러 → packages.txt에 시스템 패키지 확인

에러가 나면 멈추지 말고 위 규칙대로 알아서 해결하고 계속 진행해.
모든 단계가 끝나면 최종 상태를 요약해서 알려줘.
