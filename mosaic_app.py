"""
MOSAIC — Multimodal Online Source Authenticity and Integrity Checker
Run with: streamlit run mosaic_app_new.py

Authors: [Team MOSAIC]
Date: 2025
Description: Main Streamlit app for the MOSAIC fake news detection system.
             Uses an ensemble of 12 expert models to classify news as Real/Fake.
"""

import os
import sys
import tempfile
import streamlit as st
from PIL import Image

# Add the MoPeD submodule to path so we can import from it
# Keep in mind this will change depending on your project structure and where you run the app from
sys.path.append(os.path.join(os.getcwd(), "MoPeD"))

# Basic page config - wide layout looks better for our two-column design
st.set_page_config(
    page_title="MOSAIC — Fake News Detector",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# CSS STYLING
# Uses custom fonts from Google Fonts (Space Mono + Syne) for the sci-fi look.
# Most of the dark theme colours are hardcoded here since Streamlit's theming
# system doesn't give us enough control over individual components.

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;600;700;800&display=swap');

html, body, [class*="css"] { font-family: 'Syne', sans-serif; }
#MainMenu, footer, header   { visibility: hidden; }
.block-container            { padding: 2rem 3rem 4rem 3rem; max-width: 1200px; }

/* ---------- Header ---------- */
.mosaic-header   { text-align: center; padding: 2rem 0; }
.mosaic-title    { font-family:'Syne',sans-serif; font-weight:800; font-size:3.5rem;
                   letter-spacing:0.4em; color:#00d4ff; margin:0; }
.mosaic-subtitle { font-family:'Space Mono',monospace; font-size:0.7rem;
                   letter-spacing:0.25em; color:#8888a8; text-transform:uppercase; }
.mosaic-divider  { width:80px; height:1px; background:#1e1e2e; margin:2rem auto; }

/* ---------- Banner shown while models are loading ---------- */
.load-banner {
    font-family:'Space Mono',monospace; font-size:0.62rem; letter-spacing:0.18em;
    color:#00d4ff; background:rgba(0,212,255,0.06); border:1px solid rgba(0,212,255,0.18);
    border-radius:4px; padding:0.5rem 1rem; text-align:center; margin-bottom:1.5rem;
    animation: pulse 2s infinite;
}
@keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.45} }

/* ---------- Section title with left cyan bar ---------- */
.card-title {
    font-family:'Space Mono',monospace; font-size:0.65rem; letter-spacing:0.2em;
    text-transform:uppercase; color:#5c5c7a; margin-bottom:1.25rem;
    border-left:2px solid #00d4ff; padding-left:0.75rem;
}

/* ---------- All input labels ---------- */
.stTextArea label, .stTextInput label,
.stFileUploader label, .stSlider label, .stToggle label {
    font-family:'Space Mono',monospace !important; font-size:0.62rem !important;
    letter-spacing:0.18em !important; text-transform:uppercase !important;
    color:#5c5c7a !important;
}

/* ---------- Tab bar ---------- */
.stTabs [data-baseweb="tab-list"] {
    background:transparent !important;
    border-bottom:1px solid #1e1e2e !important;
    gap:0.5rem !important;
}
.stTabs [data-baseweb="tab"] {
    background:transparent !important;
    font-family:'Space Mono',monospace !important;
    font-size:0.65rem !important; letter-spacing:0.15em !important;
    text-transform:uppercase !important; border:none !important;
    padding:0.6rem 1.25rem !important;
}

/* ---------- Buttons ---------- */
div.stButton > button {
    background:#00d4ff !important; border:none !important; color:#0d0d12 !important;
    font-weight:700 !important; font-family:'Space Mono',monospace !important;
    font-size:0.7rem !important; letter-spacing:0.15em !important;
    text-transform:uppercase !important; border-radius:4px !important;
    padding:0.6rem 1.5rem !important; transition:opacity 0.15s, transform 0.1s !important;
}
div.stButton > button:hover  { opacity:0.88 !important; transform:translateY(-1px) !important; }
div.stButton > button:active { transform:translateY(0) !important; }

/* ---------- Verdict box ---------- */
.verdict-container { text-align:center; padding:2rem; border-radius:8px; margin:1rem 0; }
.verdict-real      { background:rgba(0,212,132,0.05); border:1px solid #00d084; }
.verdict-fake      { background:rgba(255,85,85,0.05);  border:1px solid #ff5555; }
.verdict-uncertain { background:rgba(245,197,24,0.05); border:1px solid #f5c518; }
.verdict-text      { font-size:2.5rem; font-weight:800; line-height:1; }
.v-real { color:#00d084; } .v-fake { color:#ff5555; } .v-unc { color:#f5c518; }

/* ---------- Confidence bar ---------- */
.conf-bar-bg   { background:#1e1e2e; border-radius:4px; height:6px; margin:0.4rem auto;
                 overflow:hidden; max-width:200px; }
.conf-bar-real { height:100%; border-radius:4px; background:linear-gradient(90deg,#00875a,#00d084); }
.conf-bar-fake { height:100%; border-radius:4px; background:linear-gradient(90deg,#8b0000,#ff5555); }
.conf-bar-unc  { height:100%; border-radius:4px; background:linear-gradient(90deg,#7a6000,#f5c518); }

/* ---------- Metric pills below the verdict ---------- */
.metric-row  { display:flex; gap:0.6rem; flex-wrap:wrap; margin-top:1rem; }
.metric-pill { background:#11111a; border:1px solid #1e1e2e; border-radius:6px;
               padding:0.45rem 0.8rem; flex:1; min-width:80px; text-align:center; }
.mp-label    { font-family:'Space Mono',monospace; font-size:0.55rem; letter-spacing:0.12em;
               text-transform:uppercase; color:#5c5c7a; display:block; }
.mp-value    { font-size:1rem; font-weight:700; color:#e8e8f0; display:block; margin-top:0.15rem; }

/* ---------- URL placeholder box ---------- */
.url-placeholder {
    background:#0f0f1a; border:1px dashed #2a2a4a; border-radius:8px;
    padding:1.1rem; text-align:center; color:#e8e8f0;
    font-family:'Space Mono',monospace; font-size:0.63rem; line-height:1.8;
}

/* ---------- Per-expert rows ---------- */
.expert-row {
    display:flex; align-items:center; padding:0.5rem 0.75rem;
    background:#11111a; border:1px solid #1e1e2e; border-radius:4px;
    margin-bottom:0.3rem; gap:1rem; transition:border-color 0.15s;
}
.expert-row:hover { border-color:#2e2e4e; }
.expert-name      { font-family:'Space Mono',monospace; font-size:0.65rem;
                    color:#a0a0b8; min-width:160px; }

/* ---------- st.metric overrides ---------- */
[data-testid="stMetricValue"] { color:#00d4ff !important; font-family:'Space Mono',monospace !important; }
[data-testid="stMetricLabel"] { color:#5c5c7a !important; font-family:'Space Mono',monospace !important;
                                 font-size:0.6rem !important; letter-spacing:0.1em !important;
                                 text-transform:uppercase !important; }
</style>
""", unsafe_allow_html=True)

# MODEL LOADING
# We cache the ModelManager so it only loads once per server session.
# The @st.cache_resource decorator keeps it alive across reruns.

@st.cache_resource(show_spinner=False)
def load_manager():
    from model_manager import ModelManager
    return ModelManager()

# HELPER FUNCTIONS
# Separated out to keep the main layout code cleaner

def render_verdict(result: dict):
    """
    Renders the main verdict box with confidence bar and metric pills.
    result dict should have keys: final_verdict, overall_confidence,
    vote_strength, agreement, language_detected, used_real_image, most_confident_model
    """
    v        = result["final_verdict"]
    conf     = result["overall_confidence"]
    strength = result["vote_strength"]
    agree    = result["agreement"]
    lang     = result["language_detected"]
    has_img  = result["used_real_image"]
    best     = result["most_confident_model"]
    reason   = result.get("uncertainty_reason") or ""

    # Map verdict string to the right CSS class names
    cls_map = {"Real": "verdict-real", "Fake": "verdict-fake", "Uncertain": "verdict-uncertain"}
    txt_map = {"Real": "v-real",       "Fake": "v-fake",       "Uncertain": "v-unc"}
    bar_map = {"Real": "conf-bar-real","Fake": "conf-bar-fake","Uncertain": "conf-bar-unc"}

    # Only show the uncertainty reason line if verdict is Uncertain
    unc_line = ""
    if (v == "Uncertain"):
        unc_line = (
            f'<div style="font-family:Space Mono;font-size:0.6rem;color:#b59a00;margin-top:0.6rem;">'
            f'REASON: {reason.replace("_", " ").upper()}</div>'
        )

    # Display language as ZH or EN (could extend to other languages later)
    lang_display = "ZH" if (lang == "zh") else "EN"

    st.markdown(f"""
    <div class="verdict-container {cls_map.get(v, 'verdict-uncertain')}">
        <div style="font-family:Space Mono;font-size:0.6rem;color:#8888a8;margin-bottom:0.5rem;">VERDICT</div>
        <div class="verdict-text {txt_map.get(v, 'v-unc')}">{v.upper()}</div>
        <div style="font-family:Space Mono;font-size:0.6rem;color:#8888a8;margin-top:1rem;">
            CONFIDENCE {conf:.0%}
        </div>
        <div class="conf-bar-bg">
            <div class="{bar_map.get(v, 'conf-bar-unc')}" style="width:{conf * 100:.1f}%"></div>
        </div>
        {unc_line}
    </div>
    <div class="metric-row">
        <div class="metric-pill">
            <span class="mp-label">Vote Strength</span>
            <span class="mp-value">{strength:.2f}</span>
        </div>
        <div class="metric-pill">
            <span class="mp-label">Agreement</span>
            <span class="mp-value">{agree:.0%}</span>
        </div>
        <div class="metric-pill">
            <span class="mp-label">Language</span>
            <span class="mp-value">{lang_display}</span>
        </div>
        <div class="metric-pill">
            <span class="mp-label">Image</span>
            <span class="mp-value">{"YES" if has_img else "NO"}</span>
        </div>
    </div>
    <div style="font-family:Space Mono;font-size:0.58rem;color:#e8e8f0;text-align:center;margin-top:0.75rem;">
        MOST CONFIDENT EXPERT: {best}
    </div>
    """, unsafe_allow_html=True)


def render_expert_table(all_scores: list):
    """
    Renders the per-expert breakdown table inside the expander.
    Opacity is scaled by weight so low-weight models appear faded.
    """
    st.markdown('<div class="card-title">EXPERT BREAKDOWN</div>', unsafe_allow_html=True)

    for row in all_scores:
        # Green for Real, red for Fake
        color = "#00d084" if (row["predicted_label"] == "Real") else "#ff5555"

        # Scale opacity by weight so bad/low-confidence models look faded
        if (row["weight"] > 0):
            opacity = max(0.25, min(1.0, row["weight"] * 2))
        else:
            opacity = 0.2

        st.markdown(f"""
        <div class="expert-row" style="opacity:{opacity:.2f}">
            <span class="expert-name">{row['model']}</span>
            <span style="color:{color};font-family:Space Mono;font-size:0.7rem;font-weight:700;min-width:55px;">
                {row['predicted_label']}
            </span>
            <span style="font-family:Space Mono;font-size:0.6rem;color:#4a4a6a;">
                R={row['Real']:.2f} F={row['Fake']:.2f}
            </span>
            <span style="font-family:Space Mono;font-size:0.6rem;color:#4a4a6a;margin-left:auto;">
                w={row['weight']:.2f}
            </span>
        </div>""", unsafe_allow_html=True)


# PAGE HEADER

st.markdown("""
<div class="mosaic-header">
    <div class="mosaic-title">MOSAIC</div>
    <div class="mosaic-subtitle">Multimodal Online Source Authenticity Checker</div>
</div>
<div class="mosaic-divider"></div>
""", unsafe_allow_html=True)

# MODEL INIT
# Load models on first visit. After that load_manager() returns instantly
# from the Streamlit cache. We track the first load using session_state.

if ("manager_loaded" not in st.session_state):
    # Show a text-only loading banner (no spinner circle)
    loading_placeholder = st.empty()
    loading_placeholder.markdown(
        '<div class="load-banner">INITIALIZING OUR EXPERT MODELS...</div>',
        unsafe_allow_html=True,
    )
    try:
        manager = load_manager()
        st.session_state["manager_loaded"] = True
        # Clear the banner then rerun so the page shows the "ready" state cleanly
        loading_placeholder.empty()
        st.rerun()
    except Exception as e:
        st.error(f"Model load failed: {e}")
        st.stop()
else:
    # Models already cached — grab reference instantly
    manager = load_manager()
    st.markdown(
        '<div class="load-banner" style="color:#00d084;border-color:#00d084;animation:none;">'
        '✓ 12 EXPERT MODELS READY</div>',
        unsafe_allow_html=True,
    )

# TABS

tab_analyze, tab_benchmark, tab_diagnose = st.tabs(["Analysis", "Benchmark", "Diagnostics"])

# TAB 1 — ANALYSIS
# Main input/output tab. User pastes text + uploads image, clicks Run Analysis.

with tab_analyze:
    st.markdown("<br>", unsafe_allow_html=True)
    col_in, col_res = st.columns([1, 1], gap="large")

    with col_in:
        st.markdown('<div class="card-title">INPUT</div>', unsafe_allow_html=True)

        text_input = st.text_area(
            "Post Content",
            placeholder="Paste the text or headline here...",
            height=150,
            label_visibility="collapsed",
        )

        # Two columns: image upload on left, URL scraper placeholder on right
        img_col, url_col = st.columns(2)

        with img_col:
            up_file = st.file_uploader("Upload Image (required)", type=["jpg", "png", "webp", "jpeg"])

        with url_col:
            # TODO: teammate's scraper integration goes here
            # When scraper module is ready, replace this block with:
            #   from scraper import scrape
            #   url_input = st.text_input("Or paste a URL")
            #   if (url_input):
            #       scraped    = scrape(url_input)
            #       text_input = scraped["text"]
            #       img_path   = scraped["image_path"]
            st.markdown("""
            <div class="url-placeholder" style="margin-top:1.75rem;">
                <div style="font-size:1.1rem;margin-bottom:0.3rem;">LINK</div>
                <div>URL SCRAPING</div>
                <span style="font-size:0.6rem;">Teammate integration<br>in progress</span>
            </div>
            """, unsafe_allow_html=True)

        # Preview the uploaded image so the user can confirm it's the right one
        pil_image = None
        if (up_file is not None):
            pil_image = Image.open(up_file).convert("RGB")
            st.image(pil_image, use_container_width=True)

        st.markdown("<div style='margin-top:0.75rem'></div>", unsafe_allow_html=True)
        run_analysis = st.button("RUN ANALYSIS", use_container_width=True)

    with col_res:
        if (run_analysis):
            # Validate inputs before running the ensemble
            if (not text_input.strip()):
                st.warning("Please provide text input.")
            elif (up_file is None):
                st.warning("Please upload an image — this system requires both text and image.")
            else:
                with st.spinner("Ensemble voting in progress..."):
                    # Save the PIL image to a temp file so model_manager can read by path
                    img_path = None
                    if (pil_image is not None):
                        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
                        pil_image.save(tmp.name)
                        img_path = tmp.name

                    res = manager.get_prediction(text_input, img_path)

                st.markdown('<div class="card-title">RESULT</div>', unsafe_allow_html=True)
                render_verdict(res)

                with st.expander("Per-Expert Breakdown"):
                    render_expert_table(res["all_scores"])
        else:
            # Empty state placeholder — shown before the user clicks Run
            st.markdown("""
            <div style="height:300px;display:flex;align-items:center;justify-content:center;
                        color:#e8e8f0;font-family:Space Mono;font-size:0.7rem;letter-spacing:0.15em;
                        border:1px dashed #1e1e2e;border-radius:8px;">
                AWAITING INPUT
            </div>""", unsafe_allow_html=True)


# TAB 2 — BENCHMARK
# Runs the full model ensemble over a CSV test set and reports accuracy metrics.

with tab_benchmark:
    st.markdown("<br>", unsafe_allow_html=True)
    bcol1, bcol2 = st.columns([1, 2], gap="large")

    with bcol1:
        st.markdown('<div class="card-title">SETTINGS</div>', unsafe_allow_html=True)

        csv_path   = st.text_input("CSV Path",   value="multimodal_dataset/test_final.csv")
        image_root = st.text_input("Image Root", value="multimodal_dataset")
        n_cases    = st.slider("Number of cases", 10, 100, 30, 10)
        use_images = st.toggle("Use real images", value=True)
        run_bench  = st.button("RUN BENCHMARK", use_container_width=True)

    with bcol2:
        st.markdown('<div class="card-title">RESULTS</div>', unsafe_allow_html=True)

        if (run_bench):
            if (not os.path.exists(csv_path)):
                st.error(f"CSV not found: {csv_path}")
            else:
                with st.spinner(f"Running {n_cases} cases..."):
                    try:
                        from benchmark_loader import load_cases_from_csv

                        cases   = load_cases_from_csv(csv_path, image_root=image_root,
                                                       n=n_cases, use_images=use_images)
                        results = manager.benchmark_prompts(cases)

                        # Calculate overall and per-class accuracy
                        correct   = sum(1 for r in results if (r.get("correct") is True))
                        uncertain = sum(1 for r in results if (r["predicted"] == "Uncertain"))
                        decided   = len(results) - uncertain
                        overall   = correct / decided if (decided > 0) else 0

                        # Split by class to check for bias toward one label
                        real_sub = [r for r in results if (r["expected"] == "Real" and r["predicted"] != "Uncertain")]
                        fake_sub = [r for r in results if (r["expected"] == "Fake" and r["predicted"] != "Uncertain")]
                        real_acc = sum(1 for r in real_sub if r.get("correct")) / len(real_sub) if (len(real_sub) > 0) else 0
                        fake_acc = sum(1 for r in fake_sub if r.get("correct")) / len(fake_sub) if (len(fake_sub) > 0) else 0

                        # Show the five summary metrics at the top
                        m1, m2, m3, m4, m5 = st.columns(5)
                        m1.metric("Overall",   f"{overall:.0%}")
                        m2.metric("Real acc",  f"{real_acc:.0%}")
                        m3.metric("Fake acc",  f"{fake_acc:.0%}")
                        m4.metric("Decided",   f"{decided}/{len(results)}")
                        m5.metric("Uncertain", str(uncertain))

                        st.markdown("<div style='margin:0.75rem 0'></div>", unsafe_allow_html=True)

                        # Render each result as a colour-coded row
                        for row in results:
                            pred       = row["predicted"]
                            expected   = row.get("expected", "")
                            is_correct = row.get("correct")
                            conf       = row["confidence"]
                            text       = row["text"][:75]

                            # Green = correct, red = wrong, yellow = uncertain
                            if (is_correct is True):
                                sc = "#00d084"
                            elif (is_correct is False):
                                sc = "#ff5555"
                            else:
                                sc = "#f5c518"

                            exp_str = f"[{expected}->{pred}]" if (expected) else f"[{pred}]"

                            st.markdown(f"""
                            <div class="expert-row" style="border-left:3px solid {sc}">
                                <span style="font-family:Space Mono;font-size:0.62rem;color:#e8e8f0;min-width:120px;">
                                    {exp_str}
                                </span>
                                <span style="font-family:Space Mono;font-size:0.62rem;color:#8888a8;flex:1;">
                                    {text}
                                </span>
                                <span style="font-family:Space Mono;font-size:0.62rem;color:#5c5c7a;min-width:45px;text-align:right;">
                                    {conf:.0%}
                                </span>
                            </div>""", unsafe_allow_html=True)

                    except Exception as e:
                        st.error(f"Benchmark error: {e}")
        else:
            st.markdown("""
            <div style="height:200px;display:flex;align-items:center;justify-content:center;
                        color:#e8e8f0;font-family:Space Mono;font-size:0.7rem;
                        border:1px dashed #1e1e2e;border-radius:8px;">
                READY FOR TEST SUITE
            </div>""", unsafe_allow_html=True)


# TAB 3 — DIAGNOSTICS
# Runs two known anchor cases (one Fake, one Real) through each expert individually.
# Useful for debugging which models have collapsed or have flipped label order.

with tab_diagnose:
    st.markdown("<br>", unsafe_allow_html=True)

    # Short explanation of what this tab is for
    st.markdown("""
    <div style="font-family:'Space Mono',monospace;font-size:0.65rem;color:#e8e8f0;
                letter-spacing:0.08em;margin-bottom:1.5rem;line-height:1.9;">
        Shows raw per-expert outputs before weighting or label correction.<br>
        A healthy model should predict differently for the two anchor cases.<br>
        If a model outputs the same class for both inputs, it has collapsed and should be zeroed out.
    </div>
    """, unsafe_allow_html=True)

    dcol1, dcol2 = st.columns([1, 2], gap="large")

    with dcol1:
        st.markdown('<div class="card-title">ANCHOR CASES</div>', unsafe_allow_html=True)

        fake_anchor = st.text_area(
            "Known FAKE",
            value="hitler tests his new weapon of mass destruction circa colourized",
            height=100,
        )
        real_anchor = st.text_area(
            "Known REAL",
            value="galloping mini pony",
            height=100,
        )
        run_diag = st.button("RUN DIAGNOSE", use_container_width=True)

    with dcol2:
        st.markdown('<div class="card-title">RAW EXPERT OUTPUTS</div>', unsafe_allow_html=True)

        if (run_diag):
            # Build a flat dict of all experts with readable "Model (dataset)" labels
            all_experts = {}
            for name, expert in manager.moped_experts.items():
                all_experts[f"MoPeD ({name})"] = expert
            for name, expert in manager.coolant_experts.items():
                all_experts[f"COOLANT ({name})"] = expert
            for name, expert in manager.emaf_experts.items():
                all_experts[f"EMAF ({name})"] = expert

            # Use a blank grey image since we're only testing the text anchors
            blank_img = Image.new("RGB", (224, 224), color=(128, 128, 128))

            with st.spinner("Probing all experts..."):
                for anchor_text, anchor_label in [(fake_anchor, "FAKE"), (real_anchor, "REAL")]:
                    hc = "#ff5555" if (anchor_label == "FAKE") else "#00d084"

                    st.markdown(
                        f'<div style="color:{hc};font-family:Space Mono;font-size:0.65rem;'
                        f'letter-spacing:0.15em;margin:1.25rem 0 0.5rem 0;">'
                        f'KNOWN {anchor_label}: "{anchor_text[:65]}"</div>',
                        unsafe_allow_html=True,
                    )

                    for label, expert in all_experts.items():
                        try:
                            res      = expert.predict(anchor_text, blank_img)
                            r_prob   = float(res["Real"])
                            f_prob   = float(res["Fake"])
                            raw_pred = "Fake" if (f_prob > r_prob) else "Real"

                            # Apply the label-order correction stored in model_manager
                            order        = manager.label_order_map.get(label, "RF")
                            r_adj, f_adj = manager._apply_label_order(label, r_prob, f_prob)
                            final_pred   = "Fake" if (f_adj > r_adj) else "Real"

                            weight  = manager.expert_reliability.get(label, 1.0)
                            opacity = max(0.25, min(1.0, weight * 2)) if (weight > 0) else 0.2

                            raw_c   = "#ff5555" if (raw_pred   == "Fake") else "#00d084"
                            final_c = "#ff5555" if (final_pred == "Fake") else "#00d084"

                            st.markdown(f"""
                            <div class="expert-row" style="opacity:{opacity:.2f}">
                                <span class="expert-name">{label}</span>
                                <span style="font-family:Space Mono;font-size:0.6rem;color:#4a4a6a;min-width:130px;">
                                    [{r_prob:.3f} | {f_prob:.3f}]
                                </span>
                                <span style="color:{raw_c};font-family:Space Mono;font-size:0.65rem;min-width:55px;">
                                    {raw_pred}
                                </span>
                                <span style="font-family:Space Mono;font-size:0.6rem;color:#4a4a6a;min-width:35px;">
                                    {order}
                                </span>
                                <span style="color:{final_c};font-family:Space Mono;font-size:0.65rem;
                                            font-weight:700;min-width:55px;">
                                    {final_pred}
                                </span>
                                <span style="font-family:Space Mono;font-size:0.58rem;color:#4a4a6a;margin-left:auto;">
                                    w={weight:.2f}
                                </span>
                            </div>""", unsafe_allow_html=True)

                        except Exception as e:
                            # Show a faded error row instead of crashing the whole tab
                            st.markdown(f"""
                            <div class="expert-row" style="opacity:0.3">
                                <span class="expert-name">{label}</span>
                                <span style="color:#ff5555;font-family:Space Mono;font-size:0.62rem;">
                                    ERROR: {str(e)[:70]}
                                </span>
                            </div>""", unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="height:250px;display:flex;align-items:center;justify-content:center;
                        color:#e8e8f0;font-family:Space Mono;font-size:0.7rem;
                        border:1px dashed #1e1e2e;border-radius:8px;">
                READY FOR DIAGNOSTICS
            </div>""", unsafe_allow_html=True)


# ==============================================================================
# FOOTER
# ==============================================================================
st.markdown("""
<div style="text-align:center;margin-top:4rem;padding-top:2rem;border-top:1px solid #1e1e2e;
            font-family:'Space Mono',monospace;font-size:0.58rem;letter-spacing:0.15em;color:#e8e8f0;">
    MOSAIC — AI-POWERED MULTIMODAL FAKE NEWS DETECTION &nbsp;|&nbsp;
    12 EXPERT ENSEMBLE &nbsp;|&nbsp; WEIGHTED VOTING
</div>
""", unsafe_allow_html=True)