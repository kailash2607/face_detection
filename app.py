import base64
from html import escape

import cv2
import joblib
import numpy as np
import streamlit as st

from face_utils import analyse_image, predict_probabilities

st.set_page_config(page_title="Cricketer Face ID", page_icon="🏏", layout="wide")

MODEL_FILE = "face_model.joblib"

# ---------- Styling (Cricket Stadium & Pitch Theme) ----------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@500;600;700;800;900&family=JetBrains+Mono:wght@500;700&display=swap');

.stApp {
    background: #041009;
    font-family: 'Outfit', sans-serif;
    color: #E2E8F0;
}

/* Stadium lights background effect */
.stApp::before {
    content: "";
    position: fixed;
    inset: 0;
    pointer-events: none;
    z-index: 0;
    background: 
        radial-gradient(700px circle at 50% 0%, rgba(16, 185, 129, 0.22), transparent 70%),
        radial-gradient(500px circle at 15% 20%, rgba(220, 38, 38, 0.12), transparent 60%),
        radial-gradient(600px circle at 85% 80%, rgba(52, 211, 153, 0.15), transparent 70%);
}

#MainMenu, footer {visibility: hidden;}
[data-testid="stHeader"] {background: transparent;}
.block-container {position: relative; z-index: 1; padding-top: 1.5rem; padding-bottom: 5rem; max-width: 1100px;}
[data-testid="stMarkdownContainer"] p, [data-testid="stMarkdownContainer"] li,
[data-testid="stMarkdownContainer"] h5 {color: #E2E8F0;}
[data-testid="stCaptionContainer"] * {color: #6EE7B7 !important;}

/* Hero Banner - Pitch Card */
.hero {
    position: relative;
    padding: 2.2rem 2.5rem;
    border-radius: 20px;
    margin-bottom: 1.5rem;
    background: linear-gradient(135deg, rgba(6, 30, 18, 0.9), rgba(4, 16, 9, 0.95));
    border: 1px solid rgba(52, 211, 153, 0.35);
    box-shadow: 0 12px 40px rgba(0, 0, 0, 0.6), inset 0 1px 0 rgba(255, 255, 255, 0.1);
}
.hero-eyebrow {
    font-family: 'JetBrains Mono', monospace;
    font-size: .78rem;
    letter-spacing: .18em;
    color: #34D399;
    margin: 0 0 .5rem 0;
    font-weight: 700;
}
.hero-eyebrow::before {
    content: "";
    display: inline-block;
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: #34D399;
    margin-right: 8px;
    box-shadow: 0 0 10px #34D399;
}
.hero .hero-title {
    font-size: 3rem !important;
    font-weight: 900 !important;
    line-height: 1.1 !important;
    margin: 0 !important;
    color: #FFFFFF !important;
    text-shadow: 0 0 25px rgba(52, 211, 153, 0.25);
    letter-spacing: -0.5px;
}
.hero-sub {
    font-family: 'JetBrains Mono', monospace;
    font-size: .92rem;
    color: #A7F3D0;
    margin: .8rem 0 0 0;
}
.hero-badges span {
    display: inline-block;
    padding: 6px 16px;
    margin: 16px 8px 0 0;
    border-radius: 8px;
    font-size: .78rem;
    font-weight: 700;
    color: #F8FAFC;
    background: rgba(16, 185, 129, 0.15);
    border: 1px solid rgba(52, 211, 153, 0.4);
}

/* Upload Dropzone - Boundary Line feel */
[data-testid="stFileUploaderDropzone"] {
    border: 2px dashed rgba(52, 211, 153, 0.5) !important;
    border-radius: 16px !important;
    background: rgba(6, 30, 18, 0.6) !important;
    padding: 2.2rem 1rem !important;
    transition: .2s;
}
[data-testid="stFileUploaderDropzone"]:hover {
    border-color: #34D399 !important;
    background: rgba(16, 185, 129, 0.1) !important;
    box-shadow: 0 0 20px rgba(52, 211, 153, 0.2);
}
[data-testid="stFileUploaderDropzone"] * {color: #E2E8F0 !important;}
[data-testid="stFileUploaderDropzone"] button {
    border: 1px solid #34D399 !important;
    border-radius: 8px !important;
    color: #041009 !important;
    background: #34D399 !important;
    font-weight: 700 !important;
}

/* Cards & Match Scorecard Elements */
.card {
    background: rgba(6, 30, 18, 0.75);
    border: 1px solid rgba(52, 211, 153, 0.2);
    border-radius: 16px;
    padding: 1.5rem;
    backdrop-filter: blur(12px);
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
}
.imgcard img {width: 100%; border-radius: 12px; border: 2px solid rgba(52, 211, 153, 0.3);}
.imgcard p {font-size: .8rem; color: #6EE7B7; margin: .5rem 0 0 0; text-align: center; font-family: 'JetBrains Mono', monospace;}
.crop img {width: 128px; border-radius: 10px; border: 2px solid #34D399; display: block; margin: 0 auto; box-shadow: 0 0 15px rgba(52, 211, 153, 0.3);}

.result-label {font-family: 'JetBrains Mono', monospace; font-size: .75rem; letter-spacing: .15em; color: #34D399; font-weight: 700;}
.result-name {font-size: 2.4rem; font-weight: 900; color: #FFFFFF; margin: .2rem 0 .8rem 0; line-height: 1.15;}
.result-row {display: flex; align-items: center; gap: 1.3rem;}

/* Confidence Gauge Ring */
.ring {
    width: 96px; height: 96px; min-width: 96px; border-radius: 50%; position: relative;
    background: conic-gradient(var(--c) calc(var(--p) * 1%), rgba(255, 255, 255, 0.08) 0);
    display: flex; align-items: center; justify-content: center; box-shadow: 0 0 25px var(--c);
}
.ring::before {content: ""; position: absolute; inset: 9px; background: #061E12; border-radius: 50%;}
.ring span {position: relative; font-weight: 900; font-size: 1.25rem; color: #FFFFFF; font-family: 'JetBrains Mono', monospace;}
.verdict {display: inline-block; padding: 5px 14px; border-radius: 999px; font-size: .8rem; font-weight: 700; font-family: 'JetBrains Mono', monospace;}

/* Scorecard Progress Bars */
.prob {display: grid; grid-template-columns: 150px 1fr 52px; gap: 12px; align-items: center; margin: 10px 0; font-size: .9rem; font-weight: 600;}
.prob-bar {height: 9px; border-radius: 4px; background: rgba(255, 255, 255, 0.08); overflow: hidden; border: 1px solid rgba(52, 211, 153, 0.2);}
.prob-bar div {height: 100%; border-radius: 4px; background: linear-gradient(90deg, #10B981, #34D399);}
.prob-val {font-family: 'JetBrains Mono', monospace; text-align: right; color: #34D399; font-weight: 700;}

/* Steps & Roster Badges */
.step {background: rgba(6, 30, 18, 0.6); border: 1px solid rgba(52, 211, 153, 0.2); border-radius: 14px; padding: 1.2rem; height: 100%;}
.step-num {
    display: inline-block; background: #34D399; color: #041009; width: 30px; height: 30px;
    border-radius: 8px; text-align: center; line-height: 30px; font-weight: 900; margin-bottom: 8px; font-family: 'JetBrains Mono', monospace;
}
.chip {
    display: inline-block; padding: 6px 14px; margin: 4px 6px 4px 0; border-radius: 8px; font-size: .82rem;
    font-weight: 600; color: #F8FAFC; background: rgba(16, 185, 129, 0.15); border: 1px solid rgba(52, 211, 153, 0.3);
}

/* Sidebar Styling */
[data-testid="stSidebar"] {background: #020A05; border-right: 1px solid rgba(52, 211, 153, 0.2);}
[data-testid="stSidebar"] * {color: #E2E8F0;}
[data-testid="stSidebar"] hr {border-color: rgba(52, 211, 153, 0.2) !important;}
[data-testid="stSidebar"] [role="slider"] {background-color: #34D399 !important;}
[data-testid="stSidebar"] [data-testid="stSliderThumbValue"] {color: #34D399 !important; font-weight: 700;}
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] * {color: #6EE7B7 !important; font-size: .78rem;}
.sb-brand {display: flex; align-items: center; gap: 12px; padding: .4rem 0 .8rem;}
.sb-logo {
    width: 44px; height: 44px; border-radius: 12px; font-size: 1.4rem;
    background: linear-gradient(135deg, #10B981, #059669);
    display: flex; align-items: center; justify-content: center;
    box-shadow: 0 0 20px rgba(16, 185, 129, 0.4);
}
.sb-name {font-size: 1.3rem; font-weight: 800; color: #FFFFFF !important; margin: 0; line-height: 1.2;}
.sb-tag {font-size: .78rem; color: #34D399 !important; margin: 0; font-family: 'JetBrains Mono', monospace;}
.sb-label {font-size: .72rem; font-weight: 700; letter-spacing: .15em; color: #34D399 !important; margin: 1.1rem 0 .4rem; font-family: 'JetBrains Mono', monospace;}
.sb-card {
    background: rgba(6, 30, 18, 0.5); border: 1px solid rgba(52, 211, 153, 0.25);
    border-radius: 12px; padding: .9rem 1rem; font-size: .84rem; line-height: 1.5;
}
.sb-card b {color: #FFFFFF !important;}
</style>
""", unsafe_allow_html=True)


# ---------- Helper functions ----------
@st.cache_resource
def load_bundle():
    return joblib.load(MODEL_FILE)


def pretty(name):
    """ms_dhoni -> MS Dhoni, virat_kohli -> Virat Kohli"""
    words = name.split("_")
    return " ".join(w.upper() if len(w) <= 2 else w.capitalize() for w in words)


def to_data_uri(image, max_width=900):
    h, w = image.shape[:2]
    if w > max_width:
        image = cv2.resize(image, (max_width, int(h * max_width / w)))
    ok, buffer = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return "data:image/jpeg;base64," + base64.b64encode(buffer).decode()


def annotated_photo(image, boxes, chosen):
    """Draw a thin box around secondary faces and a bright cricket-green box around the classified subject."""
    shown = image.copy()
    thickness = max(2, image.shape[1] // 250)
    for (x, y, w, h) in boxes:
        cv2.rectangle(shown, (int(x), int(y)), (int(x + w), int(y + h)), (120, 140, 130), 1)
    x, y, w, h = chosen
    # Green bounding box for the identified cricketer face
    cv2.rectangle(shown, (int(x), int(y)), (int(x + w), int(y + h)), (153, 211, 52), thickness * 2)
    return shown


# ---------- Load the trained model ----------
try:
    bundle = load_bundle()
except FileNotFoundError:
    st.error(f"Could not find '{MODEL_FILE}'. Run `python train_model.py` first, then reload this page.")
    st.stop()

players = [pretty(c) for c in bundle["classes"]]

# ---------- Sidebar ----------
with st.sidebar:
    st.markdown(
        '<div class="sb-brand"><div class="sb-logo">🏏</div>'
        '<div><p class="sb-name">Pitch Scanner</p><p class="sb-tag">MATCH DAY RECOGNITION</p></div></div>',
        unsafe_allow_html=True)

    st.markdown('<p class="sb-label">DECISION THRESHOLD</p>', unsafe_allow_html=True)
    threshold = st.slider("Minimum confidence", 0.20, 0.80, 0.40, 0.05, label_visibility="collapsed")
    st.caption("Matches below this threshold trigger an 'UNKNOWN / UNCATALOGED' call.")

    st.markdown('<p class="sb-label">ROSTER TELEMETRY</p>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="sb-card"><b>{len(players)} players</b> indexed<br>'
        f'<b>{bundle["cv_accuracy"] * 100:.0f}%</b> validation accuracy<br>'
        'HOG + PCA + SVM classifier</div>',
        unsafe_allow_html=True)
    st.caption("Trained on official cricketer dataset. Off-roster faces are flagged as Unknown.")

# ---------- Header ----------
st.markdown(
    '<div class="hero"><p class="hero-eyebrow">STADIUM BROADCAST FEED · LIVE SCANNER</p>'
    '<p class="hero-title">Cricketer Face ID</p>'
    '<p class="hero-sub">upload a photo to detect facial vectors and match against the roster.</p>'
    '<div class="hero-badges"><span>🏏 Player Recognition</span><span>📐 HOG Features</span>'
    '<span>🤖 SVM Classifier</span><span>📊 Match Confidence</span></div></div>',
    unsafe_allow_html=True)

uploaded = st.file_uploader("Upload match photo (JPG, PNG or WEBP)", type=["jpg", "jpeg", "png", "webp"])

# ---------- Result ----------
if uploaded is not None:
    image = cv2.imdecode(np.frombuffer(uploaded.getvalue(), np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        st.error("Could not read image file. Please upload a valid PNG or JPG file.")
        st.stop()

    face, boxes, chosen = analyse_image(image)

    if face is None:
        left, right = st.columns([1, 1])
        left.markdown(f'<div class="card imgcard"><img src="{to_data_uri(image)}"/><p>Uploaded Frame</p></div>',
                      unsafe_allow_html=True)
        right.markdown(
            '<div class="card"><p class="result-label">RECOGNITION RESULT</p>'
            '<p class="result-name">No Face Detected 😕</p>'
            '<p>Ensure the player is facing the camera with clear lighting. Helmet grilles, caps, or extreme angles can affect detection.</p></div>',
            unsafe_allow_html=True)
    else:
        ranked = predict_probabilities(bundle, face)
        top_name, top_p = ranked[0]
        known = top_p >= threshold

        if known:
            color, verdict_bg = ("#34D399", "rgba(52,211,153,.2)") if top_p >= 0.7 else ("#60A5FA", "rgba(96,165,250,.2)")
            verdict = "High Confidence Match" if top_p >= 0.7 else "Moderate Confidence Match"
            title, label = pretty(top_name), "IDENTIFIED CRICKETER"
        else:
            color, verdict_bg = "#EF4444", "rgba(239,68,68,.2)"
            verdict = "Uncertain / Uncataloged"
            title, label = "Unknown Player", "RECOGNITION RESULT"

        left, right = st.columns([1, 1], gap="large")
        with left:
            st.markdown(
                f'<div class="card imgcard"><img src="{to_data_uri(annotated_photo(image, boxes, chosen))}"/>'
                '<p>Green box = target face isolated for prediction</p></div>', unsafe_allow_html=True)
            st.write("")
            st.markdown(
                f'<div class="card crop"><img src="{to_data_uri(face)}"/>'
                '<p>Normalized feature input (128×128 grayscale)</p></div>', unsafe_allow_html=True)

        with right:
            bars = "".join(
                f'<div class="prob"><div>{escape(pretty(n))}</div>'
                f'<div class="prob-bar"><div style="width:{p * 100:.0f}%"></div></div>'
                f'<div class="prob-val">{p * 100:.0f}%</div></div>'
                for n, p in ranked)
            closest = "" if known else (
                f'<p style="color:#A7F3D0;margin:.2rem 0 .8rem 0;font-family:\'JetBrains Mono\',monospace;font-size:0.82rem;">Closest match: <b>{escape(pretty(top_name))}</b> '
                f'at {top_p * 100:.0f}%, which is below the required {threshold * 100:.0f}% threshold.</p>')
            st.markdown(
                '<div class="card">'
                f'<p class="result-label">{label}</p>'
                f'<p class="result-name">{escape(title)}</p>'
                '<div class="result-row">'
                f'<div class="ring" style="--p:{top_p * 100:.0f};--c:{color}"><span>{top_p * 100:.0f}%</span></div>'
                f'<div><span class="verdict" style="background:{verdict_bg};color:{color}">{verdict}</span></div></div>'
                f'{closest}'
                '<p class="result-label" style="margin-top:1.4rem">SCORECARD PROBABILITY BREAKDOWN</p>'
                f'{bars}</div>',
                unsafe_allow_html=True)
            if len(boxes) > 1:
                st.caption(f"{len(boxes)} faces detected in frame. Highest-area candidate evaluated.")
else:
    st.markdown("### Match Day Workflow")
    s1, s2, s3 = st.columns(3)
    s1.markdown('<div class="step"><div class="step-num">1</div><br><b>Upload Photo</b><br>'
                'Provide any clear image featuring a cricketer face.</div>', unsafe_allow_html=True)
    s2.markdown('<div class="step"><div class="step-num">2</div><br><b>Extract Features</b><br>'
                'Facial coordinates are detected and normalized automatically.</div>', unsafe_allow_html=True)
    s3.markdown('<div class="step"><div class="step-num">3</div><br><b>Match Roster</b><br>'
                'Classifiers evaluate probability scores across all indexed players.</div>', unsafe_allow_html=True)
    st.write("")
    st.markdown("##### 🏏 Indexed Squad Roster")
    st.markdown("".join(f'<span class="chip">{escape(p)}</span>' for p in players), unsafe_allow_html=True)