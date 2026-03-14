"""🪞 닮은꼴 연예인 찾기 - 메인 앱"""
import streamlit as st
import random
import os
import tempfile
import numpy as np
from PIL import Image

from utils.face_matcher import build_celeb_embeddings, find_lookalikes
from utils.image_processor import process_uploaded_image

# ── 페이지 설정 ──
st.set_page_config(
    page_title="닮은꼴 연예인 찾기",
    page_icon="🪞",
    layout="wide",
)

# ── 커스텀 CSS ──
st.markdown("""
<style>
    /* 결과 카드 스타일 */
    .result-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        border-radius: 16px;
        padding: 20px;
        text-align: center;
        border: 1px solid #334155;
        transition: transform 0.3s;
    }
    .result-card:hover {
        transform: translateY(-4px);
    }

    /* 1위 카드 반짝이 애니메이션 */
    .winner-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #2d1b69 50%, #5b21b6 100%);
        border: 2px solid #a78bfa;
        animation: glow 2s ease-in-out infinite alternate;
    }
    @keyframes glow {
        from { box-shadow: 0 0 5px #a78bfa, 0 0 10px #a78bfa; }
        to { box-shadow: 0 0 15px #a78bfa, 0 0 30px #7c3aed; }
    }

    /* 유사도 텍스트 */
    .similarity-text {
        font-size: 2.2rem;
        font-weight: bold;
        background: linear-gradient(90deg, #f472b6, #a78bfa);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    /* 헤더 스타일 */
    .main-title {
        text-align: center;
        font-size: 2.8rem;
        font-weight: bold;
        margin-bottom: 0;
    }
    .sub-title {
        text-align: center;
        color: #94a3b8;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }

    /* 사이드바 연예인 목록 */
    .celeb-item {
        display: flex;
        align-items: center;
        gap: 10px;
        padding: 4px 0;
    }

    /* 푸터 */
    .footer {
        text-align: center;
        color: #64748b;
        padding: 2rem 0 1rem;
        font-size: 0.85rem;
    }
</style>
""", unsafe_allow_html=True)

# ── 로딩 메시지 ──
LOADING_MESSAGES = [
    "AI가 열심히 얼굴을 분석하고 있어요...",
    "연예인 데이터베이스를 뒤지는 중...",
    "당신의 숨겨진 매력을 찾는 중...",
    "혹시 연예인 아니세요...?",
]


@st.cache_resource(show_spinner="연예인 임베딩 로딩 중...")
def load_embeddings():
    """연예인 임베딩 로드 (캐시)"""
    return build_celeb_embeddings()


# ── 헤더 ──
st.markdown('<p class="main-title">🪞 닮은꼴 연예인 찾기</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">AI가 찾아주는 나의 연예인 쌍둥이!</p>', unsafe_allow_html=True)

# ── 사이드바: 등록된 연예인 목록 ──
celeb_dir = os.path.join(os.path.dirname(__file__), "celebrities")
celeb_files = {}
if os.path.exists(celeb_dir):
    for f in sorted(os.listdir(celeb_dir)):
        if f.lower().endswith((".jpg", ".jpeg", ".png")):
            name = os.path.splitext(f)[0]
            celeb_files[name] = os.path.join(celeb_dir, f)

with st.sidebar:
    st.markdown("### ⭐ 등록된 연예인")
    st.markdown(f"총 **{len(celeb_files)}명** 등록")
    st.divider()
    for name, path in celeb_files.items():
        col1, col2 = st.columns([1, 3])
        with col1:
            try:
                st.image(path, width=40)
            except Exception:
                st.write("🖼️")
        with col2:
            st.write(name)

# ── 메인 영역 ──
st.divider()

# 셀카 업로드
uploaded_file = st.file_uploader(
    "📸 셀카를 업로드하세요",
    type=["jpg", "jpeg", "png"],
    help="정면 사진이 가장 정확한 결과를 줍니다!",
)

if uploaded_file is not None:
    # 업로드 이미지 표시
    col_upload, col_spacer = st.columns([1, 1])
    with col_upload:
        st.image(uploaded_file, caption="📷 내 사진", width=300)

    # 분석 버튼
    if st.button("분석 시작! 🔍", type="primary", use_container_width=True):
        # 이미지 전처리
        user_image = process_uploaded_image(uploaded_file)

        with st.spinner(random.choice(LOADING_MESSAGES)):
            # 임베딩 로드
            celeb_embeddings = load_embeddings()

            if not celeb_embeddings:
                st.error("등록된 연예인 데이터가 없습니다. celebrities/ 폴더를 확인해주세요.")
            else:
                # 임시 파일로 저장 후 분석
                with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                    img = Image.fromarray(user_image)
                    img.save(tmp.name)
                    tmp_path = tmp.name

                try:
                    results = find_lookalikes(tmp_path, celeb_embeddings, top_n=3)
                finally:
                    os.unlink(tmp_path)

                if not results:
                    st.warning("😅 얼굴을 감지하지 못했어요. 정면 사진으로 다시 시도해주세요!")
                else:
                    # 1위 축하 효과
                    st.balloons()

                    st.markdown("---")
                    st.markdown("### 🏆 당신의 닮은꼴 연예인 TOP 3")
                    st.markdown("")

                    # TOP 3 결과 표시
                    cols = st.columns(3)
                    medals = ["🥇", "🥈", "🥉"]
                    card_classes = ["winner-card result-card", "result-card", "result-card"]

                    for i, (col, result) in enumerate(zip(cols, results)):
                        with col:
                            card_class = card_classes[i]
                            st.markdown(f"""
                            <div class="{card_class}">
                                <p style="font-size: 1.5rem; margin-bottom: 4px;">{medals[i]} {i+1}위</p>
                            </div>
                            """, unsafe_allow_html=True)

                            # 연예인 사진
                            if os.path.exists(result["image_path"]):
                                st.image(result["image_path"], use_container_width=True)

                            # 이름 + 유사도
                            st.markdown(f"**{result['name']}**")
                            st.markdown(f'<p class="similarity-text">{result["similarity"]}%</p>', unsafe_allow_html=True)
                            st.progress(result["similarity"] / 100)

                    st.markdown("")
                    st.info("📱 이 결과를 친구에게 공유하세요!")

# ── 푸터 ──
st.markdown("---")
st.markdown("""
<div class="footer">
    Made with ❤️ by 동원<br>
    ※ 이 서비스는 재미용입니다. AI 분석 결과는 참고만 해주세요!
</div>
""", unsafe_allow_html=True)
