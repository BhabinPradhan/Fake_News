import os
import sys
import tempfile

import streamlit as st
from PIL import Image

sys.path.append(os.path.join(os.getcwd(), "MoPeD"))

st.set_page_config(
    page_title="MOSAIC — Fake News Detector",
    layout="wide",
    initial_sidebar_state="collapsed",
)


@st.cache_resource(show_spinner=False)
def load_manager():
    from model_manager import ModelManager

    manager = ModelManager()
    return manager


st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;600;700;800&display=swap');

/* Main page styles */
html, body, [class*="css"] {
    font-family: 'Syne', sans-serif;
    background-color: #03040e;
    color: #ffffff;
}

#MainMenu, footer, header { visibility: hidden; }

.block-container {
    padding: 2rem 3rem 4rem 3rem;
    max-width: 1200px;
    background: transparent;
}

/* Background grid */
body::before {
    content: '';
    position: fixed;
    inset: 0;
    background-image:
        linear-gradient(rgba(60, 120, 255, 0.07) 1px, transparent 1px),
        linear-gradient(90deg, rgba(60, 120, 255, 0.07) 1px, transparent 1px);
    background-size: 44px 44px;
    z-index: -2;
    pointer-events: none;
}

/* Header glow */
body::after {
    content: '';
    position: fixed;
    top: -15%;
    left: 50%;
    transform: translateX(-50%);
    width: 1000px;
    height: 600px;
    background: radial-gradient(ellipse at center,
        rgba(60, 80, 255, 0.25) 0%,
        rgba(80, 40, 200, 0.12) 40%,
        transparent 68%);
    z-index: -1;
    pointer-events: none;
    filter: blur(70px);
}

/* Main title area */
.mosaic-header {
    text-align: center;
    padding: 2.5rem 0 1.5rem 0;
    position: relative;
}

.mosaic-header::before {
    content: '';
    position: absolute;
    top: 50%; left: 50%;
    transform: translate(-50%, -55%);
    width: 380px; height: 380px;
    border-radius: 50%;
    background-image: conic-gradient(
        from 0deg,
        transparent 0%,
        rgba(120, 180, 255, 0.8) 18%,
        rgba(160, 100, 255, 0.5) 38%,
        transparent 58%,
        rgba(80, 140, 255, 0.4) 78%,
        transparent 100%
    );
    -webkit-mask: radial-gradient(farthest-side, transparent calc(100% - 2px), white calc(100% - 2px));
    mask: radial-gradient(farthest-side, transparent calc(100% - 2px), white calc(100% - 2px));
    animation: spin-ring 8s linear infinite;
    z-index: -1;
    pointer-events: none;
}
@keyframes spin-ring {
    from { transform: translate(-50%, -55%) rotate(0deg); }
    to   { transform: translate(-50%, -55%) rotate(360deg); }
}

.mosaic-title {
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    font-size: 4.2rem;
    letter-spacing: 0.55em;
    margin: 0;
    color: #ffffff;
    text-shadow: 0 0 40px rgba(120, 180, 255, 0.7), 0 0 80px rgba(80, 100, 255, 0.4);
}

.mosaic-subtitle {
    font-family: 'Space Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.3em;
    color: #ffffff;
    opacity: 0.6;
    text-transform: uppercase;
    margin-top: 0.6rem;
}

.mosaic-divider {
    width: 120px; height: 1px;
    background: linear-gradient(90deg, transparent, rgba(120, 180, 255, 0.6), transparent);
    margin: 1.8rem auto;
}

/* Model status banner */
.load-banner {
    font-family: 'Space Mono', monospace;
    font-size: 0.62rem;
    letter-spacing: 0.2em;
    color: #ffffff;
    background: rgba(30, 50, 120, 0.45);
    border: 1px solid rgba(80, 140, 255, 0.5);
    border-radius: 10px;
    padding: 0.7rem 1.2rem;
    text-align: center;
    margin-bottom: 1.5rem;
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    box-shadow: 0 4px 24px rgba(60, 100, 255, 0.2), inset 0 1px 0 rgba(255,255,255,0.08);
    animation: pulse 2s infinite;
}
@keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.55; } }

/* Section heading */
.card-title {
    font-family: 'Space Mono', monospace;
    font-size: 0.6rem;
    letter-spacing: 0.28em;
    text-transform: uppercase;
    color: #ffffff;
    margin-bottom: 1.25rem;
    border-left: 2px solid rgba(100, 160, 255, 0.8);
    padding-left: 0.75rem;
}

/* Keep form labels white */
.stTextArea label, .stTextInput label, .stFileUploader label,
label, .stFileUploader > label {
    font-family: 'Space Mono', monospace !important;
    font-size: 0.6rem !important;
    letter-spacing: 0.18em !important;
    text-transform: uppercase !important;
    color: #ffffff !important;
    opacity: 0.85 !important;
}

/* Text area */
.stTextArea textarea {
    background: rgba(10, 16, 50, 0.55) !important;
    border: 1px solid rgba(80, 140, 255, 0.35) !important;
    border-radius: 10px !important;
    color: #ffffff !important;
    font-family: 'Syne', sans-serif !important;
    font-size: 0.92rem !important;
    backdrop-filter: blur(10px) !important;
    box-shadow: 0 4px 20px rgba(0,0,0,0.3), inset 0 1px 0 rgba(255,255,255,0.05) !important;
}
.stTextArea textarea::placeholder { color: rgba(255,255,255,0.25) !important; }
.stTextArea textarea:focus {
    border-color: rgba(100, 160, 255, 0.7) !important;
    box-shadow: 0 0 0 2px rgba(80, 140, 255, 0.2), 0 4px 20px rgba(0,0,0,0.3) !important;
}

/* Link input */
.stTextInput input {
    background: rgba(10, 16, 50, 0.55) !important;
    border: 1px solid rgba(80, 140, 255, 0.35) !important;
    border-radius: 10px !important;
    color: #ffffff !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.72rem !important;
    backdrop-filter: blur(10px) !important;
    box-shadow: 0 4px 20px rgba(0,0,0,0.3) !important;
}
.stTextInput input::placeholder { color: rgba(255,255,255,0.25) !important; }
.stTextInput input:focus {
    border-color: rgba(100, 160, 255, 0.7) !important;
    box-shadow: 0 0 0 2px rgba(80, 140, 255, 0.2) !important;
}

/* Upload area */
.stFileUploader > div {
    background: rgba(10, 16, 50, 0.5) !important;
    border: 1px dashed rgba(80, 140, 255, 0.45) !important;
    border-radius: 10px !important;
    backdrop-filter: blur(10px) !important;
    box-shadow: 0 4px 20px rgba(0,0,0,0.25) !important;
    transition: border-color 0.2s, box-shadow 0.2s !important;
}
.stFileUploader > div:hover {
    border-color: rgba(120, 180, 255, 0.7) !important;
    box-shadow: 0 6px 30px rgba(60, 120, 255, 0.2) !important;
}

/* Keep upload text readable */
.stFileUploader p, .stFileUploader span, .stFileUploader div,
.stFileUploader small, .stFileUploader section {
    color: #ffffff !important;
    opacity: 0.85 !important;
}

/* Upload button */
.stFileUploader button {
    background: rgba(30, 60, 160, 0.6) !important;
    border: 1px solid rgba(80, 140, 255, 0.5) !important;
    color: #ffffff !important;
    border-radius: 8px !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.68rem !important;
}

/* Main action button */
div.stButton > button {
    background: linear-gradient(135deg, #2a5fff 0%, #1a35cc 100%) !important;
    border: 1px solid rgba(120, 180, 255, 0.5) !important;
    color: #ffffff !important;
    font-weight: 700 !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.7rem !important;
    letter-spacing: 0.22em !important;
    text-transform: uppercase !important;
    border-radius: 10px !important;
    padding: 0.7rem 1.8rem !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 4px 28px rgba(40, 90, 255, 0.45), inset 0 1px 0 rgba(255,255,255,0.15) !important;
}
div.stButton > button:hover {
    background: linear-gradient(135deg, #4a7fff 0%, #2a4fee 100%) !important;
    box-shadow: 0 6px 40px rgba(60, 120, 255, 0.6), inset 0 1px 0 rgba(255,255,255,0.2) !important;
    transform: translateY(-2px) !important;
}
div.stButton > button:active { transform: translateY(0) !important; }

/* Result card */
.verdict-container {
    text-align: center;
    padding: 2.2rem 2rem;
    border-radius: 16px;
    margin: 1rem 0;
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
}
.verdict-real {
    background: rgba(0, 180, 100, 0.1);
    border: 1px solid rgba(0, 230, 150, 0.5);
    box-shadow: 0 0 40px rgba(0, 220, 140, 0.12), 0 8px 32px rgba(0,0,0,0.4);
}
.verdict-fake {
    background: rgba(220, 40, 40, 0.1);
    border: 1px solid rgba(255, 90, 90, 0.5);
    box-shadow: 0 0 40px rgba(255, 80, 80, 0.12), 0 8px 32px rgba(0,0,0,0.4);
}
.verdict-uncertain {
    background: rgba(200, 150, 0, 0.1);
    border: 1px solid rgba(240, 200, 0, 0.5);
    box-shadow: 0 0 40px rgba(240, 200, 0, 0.12), 0 8px 32px rgba(0,0,0,0.4);
}

.verdict-text { font-size: 3.2rem; font-weight: 800; line-height: 1; letter-spacing: 0.06em; }
.v-real { color: #00f096; text-shadow: 0 0 30px rgba(0, 240, 150, 0.6); }
.v-fake { color: #ff4f4f; text-shadow: 0 0 30px rgba(255, 79, 79, 0.6); }
.v-unc  { color: #ffcc00; text-shadow: 0 0 30px rgba(255, 204, 0, 0.6); }

/* Confidence bar */
.conf-bar-bg {
    background: rgba(255,255,255,0.08);
    border-radius: 4px; height: 5px;
    margin: 0.5rem auto; overflow: hidden; max-width: 240px;
}
.conf-bar-real { height:100%; border-radius:4px; background:linear-gradient(90deg,#006b40,#00f096); box-shadow:0 0 12px rgba(0,240,150,0.7); }
.conf-bar-fake { height:100%; border-radius:4px; background:linear-gradient(90deg,#6b0000,#ff4f4f); box-shadow:0 0 12px rgba(255,79,79,0.7); }
.conf-bar-unc  { height:100%; border-radius:4px; background:linear-gradient(90deg,#6b5000,#ffcc00); box-shadow:0 0 12px rgba(255,204,0,0.7); }

/* Small result boxes */
.metric-row { display:flex; gap:0.6rem; flex-wrap:wrap; margin-top:1.2rem; }
.metric-pill {
    background: rgba(15, 25, 70, 0.6);
    border: 1px solid rgba(80, 140, 255, 0.3);
    border-radius: 10px; padding: 0.65rem 0.9rem;
    flex:1; min-width:80px; text-align:center;
    backdrop-filter: blur(12px);
    box-shadow: 0 4px 16px rgba(0,0,0,0.3), inset 0 1px 0 rgba(255,255,255,0.06);
    transition: border-color 0.2s, transform 0.15s, box-shadow 0.2s;
}
.metric-pill:hover {
    border-color: rgba(120, 180, 255, 0.55);
    transform: translateY(-2px);
    box-shadow: 0 8px 24px rgba(60, 120, 255, 0.2);
}
.mp-label { font-family:'Space Mono',monospace; font-size:0.52rem; letter-spacing:0.14em; text-transform:uppercase; color:rgba(255,255,255,0.5); display:block; }
.mp-value { font-size:1.05rem; font-weight:700; color:#ffffff; display:block; margin-top:0.2rem; }

/* Explanation box */
.xai-box {
    background: rgba(15, 25, 70, 0.55);
    border: 1px solid rgba(80, 140, 255, 0.28);
    border-radius: 12px; padding: 1rem 1.15rem;
    margin-top: 0.85rem;
    backdrop-filter: blur(16px);
    box-shadow: 0 4px 24px rgba(0,0,0,0.35), inset 0 1px 0 rgba(255,255,255,0.05);
}
.xai-label { font-family:'Space Mono',monospace; font-size:0.55rem; color:rgba(255,255,255,0.5); letter-spacing:0.18em; text-transform:uppercase; margin-bottom:0.5rem; }
.xai-body  { font-family:'Syne',sans-serif; font-size:0.85rem; color:#ffffff; line-height:1.75; }
.xai-footer { font-family:'Space Mono',monospace; font-size:0.65rem; color:rgba(255,255,255,0.55); margin-top:0.6rem; border-top:1px solid rgba(80,140,255,0.18); padding-top:0.55rem; }

/* Expert list */
.expert-row {
    display:flex; align-items:center; padding:0.55rem 0.9rem;
    background: rgba(12, 20, 60, 0.55);
    border: 1px solid rgba(60, 100, 220, 0.22);
    border-radius: 8px; margin-bottom:0.35rem; gap:1rem;
    backdrop-filter: blur(8px);
    box-shadow: 0 2px 12px rgba(0,0,0,0.25);
    transition: border-color 0.15s, background 0.15s, transform 0.15s;
}
.expert-row:hover {
    border-color: rgba(100, 160, 255, 0.45);
    background: rgba(18, 32, 88, 0.7);
    transform: translateX(3px);
}
.expert-name { font-family:'Space Mono',monospace; font-size:0.62rem; color:rgba(255,255,255,0.85); min-width:160px; }

/* Empty result state */
.awaiting-box {
    height:320px; display:flex; flex-direction:column;
    align-items:center; justify-content:center;
    border: 1px dashed rgba(80, 140, 255, 0.35);
    border-radius: 16px;
    background: rgba(10, 18, 55, 0.45);
    backdrop-filter: blur(16px);
    box-shadow: 0 8px 32px rgba(0,0,0,0.4), inset 0 1px 0 rgba(255,255,255,0.04);
    gap: 0.75rem;
}

/* Input hint */
.hint-text {
    font-family:'Space Mono',monospace; font-size:0.58rem;
    color:rgba(255,255,255,0.55); letter-spacing:0.1em; line-height:1.9;
    border-left:2px solid rgba(80,140,255,0.5); padding-left:0.75rem;
    margin-bottom:1rem;
}

/* Spinner color */
.stSpinner > div { border-top-color: #4a8fff !important; }

/* Scrollbar */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: #03040e; }
::-webkit-scrollbar-thumb { background: rgba(80,140,255,0.4); border-radius:4px; }

</style>
""", unsafe_allow_html=True)


# Helper functions

def render_verdict(result: dict):
    v = result["final_verdict"]
    conf = result["overall_confidence"]
    strength = result["vote_strength"]
    agree = result["agreement"]
    lang = result["language_detected"]
    has_img = result["used_real_image"]
    best = result["most_confident_model"]
    reason = result.get("uncertainty_reason") or ""

    cls = {
        "Real": "verdict-real",
        "Fake": "verdict-fake",
        "Uncertain": "verdict-uncertain",
    }
    txt = {
        "Real": "v-real",
        "Fake": "v-fake",
        "Uncertain": "v-unc",
    }
    bar = {
        "Real": "conf-bar-real",
        "Fake": "conf-bar-fake",
        "Uncertain": "conf-bar-unc",
    }

    if v == "Uncertain":
        unc_line = (
            f'<div style="font-family:Space Mono;font-size:0.58rem;color:#ffcc00;'
            f'margin-top:0.7rem;letter-spacing:0.12em;">'
            f'REASON: {reason.replace("_", " ").upper()}</div>'
        )
    else:
        unc_line = ""

    st.markdown(f"""
    <div class="verdict-container {cls.get(v,'verdict-uncertain')}">
        <div style="font-family:Space Mono;font-size:0.55rem;color:rgba(255,255,255,0.5);
                    margin-bottom:0.7rem;letter-spacing:0.3em;">VERDICT</div>
        <div class="verdict-text {txt.get(v,'v-unc')}">{v.upper()}</div>
        <div style="font-family:Space Mono;font-size:0.58rem;color:rgba(255,255,255,0.55);
                    margin-top:1.1rem;letter-spacing:0.15em;">CONFIDENCE &nbsp; {conf:.0%}</div>
        <div class="conf-bar-bg">
            <div class="{bar.get(v,'conf-bar-unc')}" style="width:{conf*100:.1f}%"></div>
        </div>
        {unc_line}
    </div>
    <div class="metric-row">
        <div class="metric-pill"><span class="mp-label">Vote Strength</span><span class="mp-value">{strength:.2f}</span></div>
        <div class="metric-pill"><span class="mp-label">Agreement</span><span class="mp-value">{agree:.0%}</span></div>
        <div class="metric-pill"><span class="mp-label">Language</span><span class="mp-value">{'ZH' if lang=='zh' else 'EN'}</span></div>
        <div class="metric-pill"><span class="mp-label">Image</span><span class="mp-value">{'YES' if has_img else 'NO'}</span></div>
    </div>
    <div style="font-family:Space Mono;font-size:0.55rem;color:rgba(255,255,255,0.35);
                text-align:center;margin-top:0.9rem;letter-spacing:0.1em;">
        MOST CONFIDENT EXPERT: {best}
    </div>
    """, unsafe_allow_html=True)


def render_expert_table(all_scores: list):
    st.markdown('<div class="card-title">EXPERT BREAKDOWN</div>', unsafe_allow_html=True)

    for row in all_scores:
        c = "#00f096" if row["predicted_label"] == "Real" else "#ff4f4f"

        if row["weight"] > 0:
            opacity = max(0.3, min(1.0, row["weight"] * 2))
        else:
            opacity = 0.25

        st.markdown(f"""
        <div class="expert-row" style="opacity:{opacity:.2f}">
            <span class="expert-name">{row['model']}</span>
            <span style="color:{c};font-family:Space Mono;font-size:0.68rem;font-weight:700;min-width:55px;">{row['predicted_label']}</span>
            <span style="font-family:Space Mono;font-size:0.58rem;color:rgba(255,255,255,0.5);">R={row['Real']:.2f} &nbsp; F={row['Fake']:.2f}</span>
            <span style="font-family:Space Mono;font-size:0.58rem;color:rgba(255,255,255,0.35);margin-left:auto;">w={row['weight']:.2f}</span>
        </div>""", unsafe_allow_html=True)


def render_xai_summary(result: dict):
    verdict = result["final_verdict"]
    all_scores = result["all_scores"]

    if verdict == "Uncertain":
        reason = result.get("uncertainty_reason", "").replace("_", " ")
        st.markdown(f"""
        <div class="xai-box">
            <div class="xai-label">WHY UNCERTAIN</div>
            <div class="xai-body">The ensemble could not reach a confident decision due to {reason}.
            This input may fall outside the training distribution of the available models.</div>
        </div>""", unsafe_allow_html=True)
        return

    supporters = sorted(
        [
            r
            for r in all_scores
            if r["predicted_label"] == verdict and r["weight"] > 0.0
        ],
        key=lambda x: x["weight"] * abs(x["Fake"] - x["Real"]),
        reverse=True,
    )[:3]

    if not supporters:
        return

    model_names = ", ".join(r["model"] for r in supporters)
    dataset_descriptions = {
        "snopes": "English political fact-checking",
        "xfacta": "cross-domain news verification",
        "weibo":  "Chinese social media",
        "mmhl":   "medical and health misinformation",
    }
    datasets_mentioned = []
    for r in supporters:
        for key, desc in dataset_descriptions.items():
            if key in r["model"].lower() and desc not in datasets_mentioned:
                datasets_mentioned.append(desc)

    if datasets_mentioned:
        dataset_str = " and ".join(datasets_mentioned)
    else:
        dataset_str = "multiple domains"

    total_active = len([r for r in all_scores if r["weight"] > 0])
    strong_voters = len(
        [
            r
            for r in all_scores
            if r["predicted_label"] == verdict
            and abs(r["Fake"] - r["Real"]) > 0.4
            and r["weight"] > 0.3
        ]
    )
    color = "#00f096" if verdict == "Real" else "#ff4f4f"

    st.markdown(f"""
    <div class="xai-box">
        <div class="xai-label">WHY {verdict.upper()}</div>
        <div class="xai-body">
            Decision driven primarily by <span style="color:{color};font-weight:700;">{model_names}</span>,
            trained on {dataset_str} datasets. These models showed the strongest and most consistent signal
            toward a <span style="color:{color};font-weight:700;">{verdict}</span> verdict.
        </div>
        <div class="xai-footer">
            {strong_voters} of {total_active} active models voted
            <span style="color:{color};font-weight:700;">{verdict}</span> with high confidence.
        </div>
    </div>""", unsafe_allow_html=True)


# Page header
st.markdown("""
<div class="mosaic-header">
    <div class="mosaic-title">MOSAIC</div>
    <div class="mosaic-subtitle">Multimodal Online Source Authenticity &amp; Integrity Checker</div>
</div>
<div class="mosaic-divider"></div>
""", unsafe_allow_html=True)


# Load the models once
if "manager_loaded" not in st.session_state:
    st.markdown('<div class="load-banner">⬡ &nbsp; INITIALISING EXPERT MODELS...</div>', unsafe_allow_html=True)
    try:
        manager = load_manager()
        st.session_state["manager_loaded"] = True
        st.rerun()
    except Exception as e:
        st.error(f"Model load failed: {e}")
        st.stop()
else:
    manager = load_manager()
    st.markdown(
        '<div class="load-banner" style="color:#00f096;border-color:rgba(0,240,150,0.45);'
        'background:rgba(0,50,25,0.45);animation:none;">⬡ &nbsp; EXPERT MODELS READY</div>',
        unsafe_allow_html=True,
    )


st.markdown("<br>", unsafe_allow_html=True)
col_in, col_res = st.columns([1, 1], gap="large")


# Left side input area
with col_in:
    st.markdown('<div class="card-title">INPUT</div>', unsafe_allow_html=True)
    st.markdown("""
        <div class="hint-text">
            BEST RESULTS: Real news headlines or social media posts with their original image.<br>
            Designed for structured misinformation — not short phrases or mismatched images.
        </div>""", unsafe_allow_html=True)

    default_text = st.session_state.pop("pending_text", st.session_state.get("scraped_text", ""))
    text_input = st.text_area(
        "Post Content", value=default_text,
        placeholder="Paste the text or headline here...",
        height=150, label_visibility="collapsed",
    )

    img_col, url_col = st.columns(2)
    with img_col:
        up_file = st.file_uploader(
            "Upload Image (required)",
            type=["jpg", "png", "webp", "jpeg"],
        )

    with url_col:
        from scraper import get_scraped_data

        url_input = st.text_input("Paste Link", placeholder="Enter URL...", key="url_input_value")
        if url_input and url_input != st.session_state.get("last_scraped_url"):
            with st.spinner("Extracting content..."):
                try:
                    scraped = get_scraped_data(url_input)
                    st.session_state["scraped_text"] = scraped["text"]
                    st.session_state["scraped_image"] = scraped["image"]
                    st.session_state["last_scraped_url"] = url_input
                    st.session_state["pending_text"] = scraped["text"]
                    st.session_state["scrape_warning"] = scraped.get("warning")
                    st.rerun()
                except Exception as e:
                    st.session_state["last_scraped_url"] = url_input
                    st.error(f"Scrape failed: {e}")

    pil_image = None
    if up_file:
        pil_image = Image.open(up_file).convert("RGB")
        st.image(pil_image, width='stretch')
    elif st.session_state.get("scraped_image"):
        pil_image = st.session_state["scraped_image"]
        st.image(pil_image, width='stretch')
        url = st.session_state.get("last_scraped_url", "")
        domain = url.split("/")[2] if url.startswith("http") else url
        st.markdown(
            f'<div style="font-family:Space Mono;font-size:0.55rem;color:rgba(255,255,255,0.45);'
            f'letter-spacing:0.12em;margin-top:0.3rem;">⬡ SOURCE: {domain}</div>',
            unsafe_allow_html=True,
        )
        if st.session_state.get("scrape_warning"):
            st.warning(st.session_state["scrape_warning"])
        if st.button("CLEAR", key="clear_scrape"):
            for k in (
                "scraped_text",
                "scraped_image",
                "last_scraped_url",
                "pending_text",
                "url_input_value",
                "scrape_warning",
            ):
                st.session_state.pop(k, None)
            st.session_state["url_input_value"] = ""
            st.rerun()

    st.markdown("<div style='margin-top:0.85rem'></div>", unsafe_allow_html=True)
    run_analysis = st.button("RUN ANALYSIS", width='stretch')


# Right side result area
with col_res:
    if run_analysis:
        if not text_input.strip():
            st.warning("Please provide text input.")
        elif not up_file and not st.session_state.get("scraped_image"):
            st.warning("Please upload an image — both text and image are required.")
        else:
            with st.spinner("Ensemble voting in progress..."):
                tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
                pil_image.save(tmp.name)
                res = manager.get_prediction(text_input, tmp.name)
            st.markdown('<div class="card-title">RESULT</div>', unsafe_allow_html=True)
            render_verdict(res)
            render_xai_summary(res)
            with st.expander("Per-Expert Breakdown"):
                render_expert_table(res["all_scores"])
    else:
        st.markdown("""
        <div class="awaiting-box">
        <div style="font-size:2rem;opacity:0.18;color:#6090ff;">⬡</div>
        <div style="font-family:Space Mono;font-size:0.7rem;letter-spacing:0.25em;
                        color:rgba(255,255,255,0.7);font-weight:700;">AWAITING INPUT</div>
            <div style="font-family:Space Mono;font-size:0.55rem;letter-spacing:0.14em;
                        color:rgba(255,255,255,0.3);">ENTER TEXT AND IMAGE TO BEGIN ANALYSIS</div>
        </div>""", unsafe_allow_html=True)


# Footer
st.markdown("""
<div style="text-align:center;margin-top:4rem;padding-top:1.5rem;
            border-top:1px solid rgba(60,100,220,0.18);
            font-family:'Space Mono',monospace;font-size:0.55rem;
            letter-spacing:0.18em;color:rgba(255,255,255,0.35);">
    MOSAIC &nbsp;—&nbsp; AI-POWERED MULTIMODAL FAKE NEWS DETECTION &nbsp;|&nbsp;
    EXPERT ENSEMBLE &nbsp;|&nbsp; WEIGHTED VOTING
</div>
""", unsafe_allow_html=True)
