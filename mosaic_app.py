"""
MOSAIC — Multimodal Online Source Authenticity and Integrity Checker
Run with: streamlit run mosaic_app.py
"""

# ── Imports ───────────────────────────────────────────────────────────────────
# os/sys   : file paths and Python path manipulation
# tempfile : save uploaded images to disk temporarily so model_manager can read them
# time     : used for the loading poll loop
# streamlit: the entire UI framework
# PIL      : open and convert uploaded images

import os
import sys
import tempfile
import time
import streamlit as st
from PIL import Image

# Adds the MoPeD folder to Python's search path so its modules can be imported
sys.path.append(os.path.join(os.getcwd(), "MoPeD"))


# ── Page config ───────────────────────────────────────────────────────────────
# Must be the very first Streamlit call in the file — sets browser tab title,
# switches to wide layout (full browser width), and hides the sidebar by default.
st.set_page_config(
    page_title="MOSAIC — Fake News Detector",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# Model loading
# @st.cache_resource tells Streamlit: run this function once, cache the result,
# and reuse the same object on every page rerun and for every user session.
@st.cache_resource(show_spinner=False)
def load_manager():
    from model_manager import ModelManager
    from benchmark_loader import load_cases_from_csv
    manager = ModelManager()
    return manager

# ── CSS ───────────────────────────────────────────────────────────────────────
# st.markdown() renders any string as HTML in the page.
# unsafe_allow_html=True is required whenever you pass real HTML/CSS 
# Everything inside <style> tags is standard CSS that overrides Streamlit's defaults.
st.markdown("""
<style>

/* Load two Google Fonts:
   - Syne      : the display/heading font — bold, geometric, modern
   - Space Mono: the monospace font used for labels, stats, and code-like text */
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;600;700;800&display=swap');

/* Apply Syne as the default font across the whole app */
html, body, [class*="css"] { font-family: 'Syne', sans-serif; }

/* Hide Streamlit's default top menu, footer, and header — cleaner demo look */
#MainMenu, footer, header { visibility: hidden; }

/* Constrain the main content width and add breathing room */
.block-container { padding: 2rem 3rem 4rem 3rem; max-width: 1200px; }

/* ── Header section ──
   Centers the MOSAIC title and subtitle on the page */
.mosaic-header  { text-align: center; padding: 2rem 0; }
.mosaic-title   {
    font-family: 'Syne', sans-serif; font-weight: 800; font-size: 3.5rem;
    letter-spacing: 0.4em; color: #00d4ff; margin: 0;
}
.mosaic-subtitle {
    font-family: 'Space Mono', monospace; font-size: 0.7rem;
    letter-spacing: 0.25em; color: #8888a8; text-transform: uppercase;
}
/* A thin horizontal rule between header and content */
.mosaic-divider { width: 80px; height: 1px; background: #1e1e2e; margin: 2rem auto; }

/* ── Loading / ready banner ──
   Thin bar at the top of the page showing model init status.
   Uses a CSS keyframe animation to pulse opacity while loading. */
.load-banner {
    font-family: 'Space Mono', monospace; font-size: 0.62rem; letter-spacing: 0.18em;
    color: #00d4ff; background: rgba(0,212,255,0.06);
    border: 1px solid rgba(0,212,255,0.18); border-radius: 4px;
    padding: 0.5rem 1rem; text-align: center; margin-bottom: 1.5rem;
    animation: pulse 2s infinite;
}
@keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.45; } }

/* ── Card title ──
   Small uppercase label with a cyan left border — used above INPUT and RESULT */
.card-title {
    font-family: 'Space Mono', monospace; font-size: 0.65rem; letter-spacing: 0.2em;
    text-transform: uppercase; color: #9c9cb5; margin-bottom: 1.25rem;
    border-left: 2px solid #00d4ff; padding-left: 0.75rem;
}

/* ── Streamlit widget label overrides ──
   Streamlit generates its own <label> elements — these selectors target them
   and apply the Space Mono font so they match the rest of the UI */
.stTextArea label, .stTextInput label, .stFileUploader label {
    font-family: 'Space Mono', monospace !important;
    font-size: 0.62rem !important; letter-spacing: 0.18em !important;
    text-transform: uppercase !important; color: #9c9cb5 !important;
}

/* ── Analyze button ──
   div.stButton > button targets Streamlit's generated button element.
   !important overrides Streamlit's inline styles which would otherwise win. */
div.stButton > button {
    background: #00d4ff !important; border: none !important; color: #0d0d12 !important;
    font-weight: 700 !important; font-family: 'Space Mono', monospace !important;
    font-size: 0.7rem !important; letter-spacing: 0.15em !important;
    text-transform: uppercase !important; border-radius: 4px !important;
    padding: 0.6rem 1.5rem !important;
    transition: opacity 0.15s, transform 0.1s !important;
}
div.stButton > button:hover  { opacity: 0.88 !important; transform: translateY(-1px) !important; }
div.stButton > button:active { transform: translateY(0) !important; }

/* ── Verdict container ──
   The big result box. Three variants (real/fake/uncertain) share the base class
   and each adds its own border colour and faint background tint.
   rgba() = red, green, blue, alpha(opacity) */
.verdict-container { text-align: center; padding: 2rem; border-radius: 8px; margin: 1rem 0; }
.verdict-real      { background: rgba(0,212,132,0.05); border: 1px solid #00d084; }
.verdict-fake      { background: rgba(255,85,85,0.05);  border: 1px solid #ff5555; }
.verdict-uncertain { background: rgba(245,197,24,0.05); border: 1px solid #f5c518; }
.verdict-text      { font-size: 2.5rem; font-weight: 800; line-height: 1; }
.v-real { color: #00d084; }
.v-fake { color: #ff5555; }
.v-unc  { color: #f5c518; }

/* ── Confidence bar ──
   Two-layer trick: the outer .conf-bar-bg is the grey track,
   the inner div (width set dynamically in Python) is the coloured fill.
   The gradient goes from a darker shade on the left to the bright colour on the right. */
.conf-bar-bg   { background: #1e1e2e; border-radius: 4px; height: 6px; margin: 0.4rem auto; overflow: hidden; max-width: 200px; }
.conf-bar-real { height: 100%; border-radius: 4px; background: linear-gradient(90deg, #00875a, #00d084); }
.conf-bar-fake { height: 100%; border-radius: 4px; background: linear-gradient(90deg, #8b0000, #ff5555); }
.conf-bar-unc  { height: 100%; border-radius: 4px; background: linear-gradient(90deg, #7a6000, #f5c518); }

/* ── Metric pills ──
   A flex row of small stat boxes (vote strength, agreement, language, image).
   flex:1 makes each pill grow equally to fill the row. */
.metric-row  { display: flex; gap: 0.6rem; flex-wrap: wrap; margin-top: 1rem; }
.metric-pill {
    background: #11111a; border: 1px solid #1e1e2e; border-radius: 6px;
    padding: 0.45rem 0.8rem; flex: 1; min-width: 80px; text-align: center;
}
.mp-label { font-family: 'Space Mono', monospace; font-size: 0.55rem; letter-spacing: 0.12em; text-transform: uppercase; color: #5c5c7a; display: block; }
.mp-value { font-size: 1rem; font-weight: 700; color: #e8e8f0; display: block; margin-top: 0.15rem; }

/* ── URL scraping placeholder ──
   Dashed border signals "coming soon" — visually distinct from the file uploader */
.url-placeholder {
    background: #0f0f1a; border: 1px dashed #2a2a4a; border-radius: 8px;
    padding: 1.1rem; text-align: center; color: #6b6b8a;
    font-family: 'Space Mono', monospace; font-size: 0.63rem; line-height: 1.8;
}

/* ── Expert breakdown rows ──
   Each model's result is rendered as a flex row.
   transition: border-color makes the hover effect animate smoothly. */
.expert-row {
    display: flex; align-items: center; padding: 0.5rem 0.75rem;
    background: #11111a; border: 1px solid #1e1e2e; border-radius: 4px;
    margin-bottom: 0.3rem; gap: 1rem; transition: border-color 0.15s;
}
.expert-row:hover { border-color: #2e2e4e; }
.expert-name { font-family: 'Space Mono', monospace; font-size: 0.65rem; color: #a0a0b8; min-width: 160px; }

/* ── Streamlit spinner colour override ── */
.stSpinner > div { border-top-color: #00d4ff !important; }

</style>
""", unsafe_allow_html=True)


# ── Render helpers ────────────────────────────────────────────────────────────
# These are plain Python functions — not Streamlit-specific.
# They take data dicts returned by model_manager and emit HTML via st.markdown().

def render_verdict(result: dict):
    """
    Unpacks the result dict from manager.get_prediction() and renders:
      - The verdict box (REAL / FAKE / UNCERTAIN) with colour coding
      - A confidence bar whose width is set dynamically via inline style
      - Four metric pills: vote strength, agreement, language, image flag
      - An uncertainty reason line if the verdict is Uncertain

    The three dicts (cls, txt, bar) act as lookup tables so we pick the right
    CSS class for each verdict without a chain of if/elif statements.
    """
    v        = result["final_verdict"]         # "Real", "Fake", or "Uncertain"
    conf     = result["overall_confidence"]    # float 0–1
    strength = result["vote_strength"]         # how far apart Real/Fake votes are
    agree    = result["agreement"]             # fraction of experts that agreed
    lang     = result["language_detected"]     # "zh" or "en"
    has_img  = result["used_real_image"]       # True if a real image was provided
    best     = result["most_confident_model"]  # name of the most decisive expert
    reason   = result.get("uncertainty_reason") or ""

    # Lookup tables: verdict string → CSS class name
    cls = {"Real": "verdict-real", "Fake": "verdict-fake", "Uncertain": "verdict-uncertain"}
    txt = {"Real": "v-real",       "Fake": "v-fake",       "Uncertain": "v-unc"}
    bar = {"Real": "conf-bar-real","Fake": "conf-bar-fake", "Uncertain": "conf-bar-unc"}

    # Only render the uncertainty reason line when verdict IS uncertain
    unc_line = (
        f'<div style="font-family:Space Mono;font-size:0.6rem;color:#b59a00;margin-top:0.6rem;">'
        f'REASON: {reason.replace("_", " ").upper()}</div>'
    ) if v == "Uncertain" else ""

    # f-string embeds Python values directly into the HTML.
    # conf*100 converts 0.85 → 85.0 for the bar width percentage.
    # :.0% formats a float as a percentage string e.g. 0.85 → "85%"
    st.markdown(f"""
    <div class="verdict-container {cls.get(v, 'verdict-uncertain')}">
        <div style="font-family:Space Mono;font-size:0.6rem;color:#8888a8;margin-bottom:0.5rem;">VERDICT</div>
        <div class="verdict-text {txt.get(v, 'v-unc')}">{v.upper()}</div>
        <div style="font-family:Space Mono;font-size:0.6rem;color:#8888a8;margin-top:1rem;">
            CONFIDENCE {conf:.0%}
        </div>
        <div class="conf-bar-bg">
            <div class="{bar.get(v, 'conf-bar-unc')}" style="width:{conf*100:.1f}%"></div>
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
            <span class="mp-value">{'ZH' if lang == 'zh' else 'EN'}</span>
        </div>
        <div class="metric-pill">
            <span class="mp-label">Image</span>
            <span class="mp-value">{'YES' if has_img else 'NO'}</span>
        </div>
    </div>
    <div style="font-family:Space Mono;font-size:0.58rem;color:#4a4a6a;text-align:center;margin-top:0.75rem;">
        MOST CONFIDENT EXPERT: {best}
    </div>
    """, unsafe_allow_html=True)


def render_expert_table(all_scores: list):
    """
    Renders one row per expert model showing:
      - Model name (e.g. "MoPeD (snopes)")
      - Its individual prediction coloured green/red
      - Raw Real/Fake probability scores
      - Its reliability weight

    Opacity is calculated from the weight so low-weight (unreliable) experts
    appear faded — visually communicating that they barely influenced the result.
    max(0.25, min(1.0, weight*2)) clamps opacity between 0.25 and 1.0.
    """
    st.markdown('<div class="card-title">EXPERT BREAKDOWN</div>', unsafe_allow_html=True)
    for row in all_scores:
        c       = "#00d084" if row["predicted_label"] == "Real" else "#ff5555"
        opacity = max(0.25, min(1.0, row["weight"] * 2)) if row["weight"] > 0 else 0.2
        st.markdown(f"""
        <div class="expert-row" style="opacity:{opacity:.2f}">
            <span class="expert-name">{row['model']}</span>
            <span style="color:{c};font-family:Space Mono;font-size:0.7rem;font-weight:700;min-width:55px;">
                {row['predicted_label']}
            </span>
            <span style="font-family:Space Mono;font-size:0.6rem;color:#4a4a6a;">
                R={row['Real']:.2f} F={row['Fake']:.2f}
            </span>
            <span style="font-family:Space Mono;font-size:0.6rem;color:#4a4a6a;margin-left:auto;">
                w={row['weight']:.2f}
            </span>
        </div>""", unsafe_allow_html=True)

# Renders a summary box explaining the verdict, referencing the top supporting experts and their training domains.
def render_xai_summary(result: dict):
    verdict    = result["final_verdict"]
    all_scores = result["all_scores"]

    if verdict == "Uncertain":
        reason = result.get("uncertainty_reason", "").replace("_", " ")
        st.markdown(f"""
        <div style="background:#11111a;border:1px solid #2a2a1a;border-radius:6px;
                    padding:0.9rem 1rem;margin-top:0.75rem;">
            <div style="font-family:Space Mono;font-size:0.58rem;color:#5c5c7a;
                        letter-spacing:0.12em;text-transform:uppercase;
                        margin-bottom:0.4rem;">WHY UNCERTAIN</div>
            <div style="font-family:Syne;font-size:0.82rem;color:#c8c87a;line-height:1.6;">
                The ensemble could not reach a confident decision due to {reason}.
                This input may fall outside the training distribution of the available models.
            </div>
        </div>
        """, unsafe_allow_html=True)
        return

    # Find top 3 models that voted for the winning verdict, sorted by weight
    supporters = sorted(
        [r for r in all_scores if r["predicted_label"] == verdict and r["weight"] > 0.0],
        key=lambda x: x["weight"] * abs(x["Fake"] - x["Real"]),
        reverse=True
    )[:3]

    if not supporters:
        return

    # Build human-readable model names for the summary
    model_names = ", ".join(r["model"] for r in supporters)

    # Map dataset names to plain descriptions
    dataset_descriptions = {
        "snopes":  "English political fact-checking",
        "xfacta":  "cross-domain news verification",
        "weibo":   "Chinese social media",
        "mmhl":    "medical and health misinformation",
    }

    # Collect which datasets the top supporters were trained on
    datasets_mentioned = []
    for r in supporters:
        for key, desc in dataset_descriptions.items():
            if key in r["model"].lower() and desc not in datasets_mentioned:
                datasets_mentioned.append(desc)

    dataset_str = " and ".join(datasets_mentioned) if datasets_mentioned else "multiple domains"

    color = "#00d084" if verdict == "Real" else "#ff5555"

    st.markdown(f"""
    <div style="background:#11111a;border:1px solid #1e2e1e;border-radius:6px;
                padding:0.9rem 1rem;margin-top:0.75rem;">
        <div style="font-family:Space Mono;font-size:0.58rem;color:#5c5c7a;
                    letter-spacing:0.12em;text-transform:uppercase;
                    margin-bottom:0.4rem;">WHY {verdict.upper()}</div>
        <div style="font-family:Syne;font-size:0.82rem;color:#c8c8d8;line-height:1.6;">
            Decision driven primarily by
            <span style="color:{color};font-weight:700;">{model_names}</span>,
            trained on {dataset_str} datasets.
            These models showed the strongest and most consistent signal
            toward a <span style="color:{color};font-weight:700;">{verdict}</span> verdict.
        </div>
    </div>
    """, unsafe_allow_html=True)

# ── Page header ───────────────────────────────────────────────────────────────
# Rendered once at the top of the page on every rerun.
st.markdown("""
<div class="mosaic-header">
    <div class="mosaic-title">MOSAIC</div>
    <div class="mosaic-subtitle">Multimodal Online Source Authenticity Checker</div>
</div>
<div class="mosaic-divider"></div>
""", unsafe_allow_html=True)


# ── Model load status banner ──────────────────────────────────────────────────
# st.session_state is a dict that persists across reruns for the same user session.
# We use it to track whether the models have finished loading.
#
# Flow:
#   1. First page load:  "manager_loaded" is not in session_state yet
#                        → show pulsing "INITIALISING" banner
#                        → call load_manager() (this blocks until done)
#                        → set session_state flag and rerun once to show READY banner
#   2. All later reruns: "manager_loaded" IS in session_state
#                        → skip loading, call load_manager() instantly (cached)
#                        → show green READY banner

if "manager_loaded" not in st.session_state:
    st.markdown(
        '<div class="load-banner">INITIALISING EXPERT MODELS...</div>',
        unsafe_allow_html=True,
    )
    try:
        manager = load_manager()
        st.session_state["manager_loaded"] = True
        st.rerun()  # triggers one clean rerun to show the READY state
    except Exception as e:
        st.error(f"Model load failed: {e}")
        st.stop()   # st.stop() halts execution of the rest of the script
else:
    manager = load_manager()  # instant — already cached, no disk I/O
    st.markdown(
        '<div class="load-banner" style="color:#00d084;border-color:#00d084;animation:none;">'
        'EXPERT MODELS READY</div>',
        unsafe_allow_html=True,
    )


# Layout has two equal columns..
# st.columns([1, 1]) splits the page into two equal-width columns
# The "with col:" syntax means everything inside that block renders in that column
# gap="large" adds horizontal spacing between the columns.
st.markdown("<br>", unsafe_allow_html=True)
col_in, col_res = st.columns([1, 1], gap="large")


# Left column: inputs
with col_in:
    st.markdown('<div class="card-title">INPUT</div>', unsafe_allow_html=True)

    # st.text_area renders a multi-line text box.
    # label_visibility="collapsed" hides the label visually (we use card-title instead).
    # The return value is whatever string the user has typed — updates live on every keypress.
    # Pre-fills the text box with scraped article text after a URL is submitted.
    # Falls back to empty string if nothing has been scraped yet.
    # Without this, the text area would stay blank even after a successful scrape.
    default_text = st.session_state.pop("pending_text", st.session_state.get("scraped_text", ""))
    text_input = st.text_area(
        "Post Content",
        value=default_text,
        placeholder="Paste the text or headline here...",
        height=150,
        label_visibility="collapsed",
    )

    # Nested columns inside the left column — splits image upload and URL placeholder side by side
    img_col, url_col = st.columns(2)

    with img_col:
        # st.file_uploader renders a drag-and-drop file input.
        # type= restricts accepted file formats.
        # Returns None if no file uploaded, or a UploadedFile object if one is.
        up_file = st.file_uploader("Upload Image (required)", type=["jpg", "png", "webp", "jpeg"])

    with url_col:
        # Only fire the scraper when the URL is new — not on every Streamlit rerun.
        # Without the session_state guard, this would re-scrape on every keypress
        # in the text area or any other widget interaction.
        from scraper import get_scraped_data
        url_input = st.text_input("Paste Link", placeholder="Enter URL...", key="url_input_value")
        if url_input and url_input != st.session_state.get("last_scraped_url"):
            with st.spinner("Extracting content..."):
                try:
                    scraped = get_scraped_data(url_input)
                    st.session_state["scraped_text"]     = scraped["text"]
                    st.session_state["scraped_image"]    = scraped["image"]
                    st.session_state["last_scraped_url"] = url_input
                    st.session_state["pending_text"]     = scraped["text"]
                    st.session_state["scrape_warning"]   = scraped.get("warning")
                    st.rerun()
                    # Rerun so the text_area and image preview pick up the new values.
                    # They render above this block in the script, so without a rerun
                    # they would stay empty until the next user interaction.
                except Exception as e:
                    # Mark the URL as attempted even on failure so it doesn't
                    # loop trying to scrape the same broken URL repeatedly.
                    st.session_state["last_scraped_url"] = url_input
                    st.error(f"Scrape failed: {e}")

    # Resolve the image to pass to the ensemble.
    # Priority: manual upload > scraped image > None (text-only fallback handled by model_manager)
    # .convert("RGB") normalises any image format (PNG with alpha, CMYK, etc.) to plain RGB
    # so the model always receives a consistent input format.
    pil_image = None
    if up_file:
        # Manual upload — open and convert to RGB, show preview
        pil_image = Image.open(up_file).convert("RGB")
        st.image(pil_image, width='stretch')
    elif st.session_state.get("scraped_image"):
        # Scraped image from URL — already a PIL RGB object from scraper.py, show preview
        pil_image = st.session_state["scraped_image"]
        st.image(pil_image, width='stretch')
        # Extract the domain from the URL and show it as a source badge
        url = st.session_state.get("last_scraped_url", "")
        domain = url.split("/")[2] if url.startswith("http") else url
        st.markdown(
            f'<div style="font-family:Space Mono,monospace;font-size:0.58rem;'
            f'color:#5c5c7a;letter-spacing:0.1em;margin-top:0.3rem;">'
            f'⬡ SOURCE: {domain}</div>',
            unsafe_allow_html=True,
        )

        if st.session_state.get("scrape_warning"):
            st.warning(st.session_state["scrape_warning"])
        # Wipe all scrape-related session state so the user can enter a fresh URL
        # without the old content bleeding through into the next analysis
        if st.button("CLEAR", key="clear_scrape"):
            for k in ("scraped_text", "scraped_image", "last_scraped_url", "pending_text", "url_input_value", "scrape_warning"):
                st.session_state.pop(k, None)
            st.session_state["url_input_value"] = ""
            st.rerun()

    st.markdown("<div style='margin-top:0.75rem'></div>", unsafe_allow_html=True)

    # st.button renders a clickable button.
    # Returns True only on the rerun immediately after the user clicks it, False otherwise.
    # use_container_width=True stretches it to fill the column width.
    run_analysis = st.button("RUN ANALYSIS", width='stretch')


# ── Right column: results ─────────────────────────────────────────────────────
with col_res:

    if run_analysis:
        # Validation: require both text and image before running
        if not text_input.strip():
            st.warning("Please provide text input.")
        elif not up_file and not st.session_state.get("scraped_image"): #  This only runs when up_file is None. If a file is uploaded, it takes priority and the scraped image logic is completely skipped.
            st.warning("Please upload an image — both text and image are required.")
        else:
            # st.spinner shows an animated loading indicator while the indented block runs.
            # Everything inside the `with` block is synchronous — the page waits here
            # until get_prediction() returns.
            with st.spinner("Ensemble voting in progress..."):

                # Save the PIL image to a temp file on disk.
                # model_manager expects a file path string, not a PIL object.
                # delete=False keeps the file alive after the `with` block exits.
                tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
                pil_image.save(tmp.name)
                img_path = tmp.name

                # This is the core call — passes text + image path to the ensemble.
                # Returns a dict with verdict, confidence, per-expert scores, etc.
                res = manager.get_prediction(text_input, img_path)

            st.markdown('<div class="card-title">RESULT</div>', unsafe_allow_html=True)

            # render_verdict() and render_expert_table() are our helper functions above —
            # they unpack the result dict and emit the HTML for the verdict box and table.
            render_verdict(res)
            render_xai_summary(res)
            # st.expander renders a collapsible section — collapsed by default.
            # Good for detail that doesn't need to be visible immediately.
            with st.expander("Per-Expert Breakdown"):
                render_expert_table(res["all_scores"])

    else:
        # Shown before the user clicks RUN ANALYSIS — just a placeholder box.
        # display:flex + align-items:center + justify-content:center = centred text
        # inside a fixed-height box.
        st.markdown("""
        <div style="height:300px;display:flex;align-items:center;justify-content:center;
                    color:#9c9cb5;font-family:Space Mono;font-size:0.7rem;letter-spacing:0.15em;
                    border:1px dashed #1e1e2e;border-radius:8px;">
            AWAITING INPUT
        </div>
        """, unsafe_allow_html=True)


# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center;margin-top:4rem;padding-top:2rem;border-top:1px solid #1e1e2e;
            font-family:'Space Mono',monospace;font-size:0.58rem;letter-spacing:0.15em;color:#9c9cb5;">
    MOSAIC — AI-POWERED MULTIMODAL FAKE NEWS DETECTION &nbsp;|&nbsp;
    EXPERT ENSEMBLE &nbsp;|&nbsp; WEIGHTED VOTING
</div>
""", unsafe_allow_html=True)