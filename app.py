"""
Semantic Entropy & Verification: modern Streamlit dashboard.
Run with: streamlit run app.py
"""

import html
import os

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from core.pipeline import SemanticEntropyPipeline

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

st.set_page_config(
    page_title="TruthCheck AI · Hallucination Detector",
    page_icon="🧠",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ───────────────────────────── Styling ─────────────────────────────
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"], .stApp { font-family: 'Inter', system-ui, sans-serif; }
.stApp { background: radial-gradient(1200px 500px at 50% -10%, #E0E7FF 0%, #F8FAFC 55%) fixed; }
#MainMenu, footer, header[data-testid="stHeader"] { visibility: hidden; height: 0; }
.block-container { max-width: 880px; padding-top: 2.2rem; padding-bottom: 4rem; }

/* Hero */
.hero { text-align: center; margin: 0.5rem 0 1.4rem; }
.hero .pill { display: inline-block; font-size: .78rem; font-weight: 600; color: #4338CA;
  background: #EEF2FF; border: 1px solid #C7D2FE; padding: .3rem .8rem; border-radius: 999px; }
.hero h1 { font-size: 2.6rem; line-height: 1.1; font-weight: 800; letter-spacing: -.03em;
  margin: .9rem 0 .5rem; color: #0F172A; }
.hero h1 span { background: linear-gradient(90deg,#4F46E5,#06B6D4); -webkit-background-clip: text;
  -webkit-text-fill-color: transparent; }
.hero p { color: #64748B; font-size: 1.05rem; max-width: 560px; margin: 0 auto; }

/* Input */
div[data-testid="stForm"] { background: #fff; border: 1px solid #E2E8F0; border-radius: 18px;
  padding: 1rem 1.1rem .4rem; box-shadow: 0 10px 30px -12px rgba(79,70,229,.25); }
div[data-testid="stForm"] input { font-size: 1.05rem; }
.stTextInput label { font-weight: 600; color: #334155; }
div.stButton > button, div[data-testid="stFormSubmitButton"] > button {
  border-radius: 12px; font-weight: 600; border: 1px solid #E2E8F0; transition: all .15s; }
div[data-testid="stFormSubmitButton"] > button { background: linear-gradient(90deg,#4F46E5,#6366F1);
  color: #fff; border: 0; padding: .55rem 1rem; }
div[data-testid="stFormSubmitButton"] > button:hover { filter: brightness(1.08); transform: translateY(-1px); }
div.stButton > button:hover { border-color: #818CF8; color: #4338CA; }

/* Answer card */
.answer { border-radius: 22px; padding: 1.6rem 1.7rem; margin: 1.4rem 0 1rem; border: 1px solid;
  box-shadow: 0 18px 40px -20px rgba(15,23,42,.35); }
.answer.good { background: linear-gradient(135deg,#ECFDF5,#fff); border-color: #A7F3D0; }
.answer.warn { background: linear-gradient(135deg,#FFFBEB,#fff); border-color: #FDE68A; }
.answer.bad  { background: linear-gradient(135deg,#FEF2F2,#fff); border-color: #FECACA; }
.answer .row { display: flex; gap: 1.4rem; align-items: center; }
.answer .body { flex: 1; min-width: 0; }
.tag { display: inline-flex; align-items: center; gap: .35rem; font-size: .8rem; font-weight: 700;
  padding: .25rem .7rem; border-radius: 999px; }
.good .tag { background: #D1FAE5; color: #047857; }
.warn .tag { background: #FEF3C7; color: #B45309; }
.bad  .tag { background: #FEE2E2; color: #B91C1C; }
.qline { color: #64748B; font-size: .92rem; margin: .7rem 0 .2rem; }
.main-answer { font-size: 1.55rem; line-height: 1.35; font-weight: 700; color: #0F172A; margin: .1rem 0 .5rem;
  letter-spacing: -.01em; word-wrap: break-word; }
.why { color: #475569; font-size: .98rem; line-height: 1.55; }
.ring { --p: 0; --c: #10B981; width: 108px; height: 108px; border-radius: 50%; flex: none;
  background: conic-gradient(var(--c) calc(var(--p) * 1%), #E2E8F0 0);
  display: grid; place-items: center; }
.ring > div { width: 84px; height: 84px; border-radius: 50%; background: #fff; display: grid;
  place-items: center; text-align: center; }
.ring b { font-size: 1.5rem; font-weight: 800; color: #0F172A; line-height: 1; }
.ring small { display: block; font-size: .65rem; color: #64748B; font-weight: 600; letter-spacing: .04em; }
.correct { margin-top: 1rem; padding: .8rem 1rem; background: #fff; border: 1px dashed #FCA5A5;
  border-radius: 12px; color: #334155; font-size: .95rem; }
.correct b { color: #B91C1C; }

/* Section + cards */
.sec { font-size: 1.15rem; font-weight: 700; color: #0F172A; margin: 2rem 0 .2rem; }
.sec-sub { color: #64748B; font-size: .93rem; margin-bottom: .8rem; }
.grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr)); gap: .8rem; }
.card { background: #fff; border: 1px solid #E2E8F0; border-radius: 16px; padding: 1rem 1.1rem; }
.card .ico { font-size: 1.3rem; }
.card .t { font-weight: 700; color: #0F172A; margin: .3rem 0 .15rem; }
.card .d { color: #64748B; font-size: .88rem; line-height: 1.45; }
.stat .v { font-size: 1.7rem; font-weight: 800; color: #0F172A; letter-spacing: -.02em; }
.stat .l { font-size: .75rem; font-weight: 700; color: #64748B; text-transform: uppercase; letter-spacing: .05em; }
.stat .d { margin-top: .3rem; }
.step { display: flex; gap: .9rem; align-items: flex-start; background: #fff; border: 1px solid #E2E8F0;
  border-radius: 14px; padding: .85rem 1rem; margin-bottom: .55rem; }
.step .n { flex: none; width: 28px; height: 28px; border-radius: 50%; background: #EEF2FF; color: #4338CA;
  font-weight: 700; display: grid; place-items: center; font-size: .85rem; }
.step .t { font-weight: 600; color: #0F172A; }
.step .d { color: #64748B; font-size: .9rem; margin-top: .1rem; }
.group { background: #fff; border: 1px solid #E2E8F0; border-radius: 14px; padding: .9rem 1.1rem; margin-bottom: .6rem; }
.group .top { display: flex; justify-content: space-between; align-items: center; gap: .6rem; }
.group .name { font-weight: 700; color: #0F172A; }
.group .pct { font-weight: 700; color: #4338CA; }
.bar { height: 8px; background: #E2E8F0; border-radius: 99px; margin: .5rem 0 .6rem; overflow: hidden; }
.bar > div { height: 100%; background: linear-gradient(90deg,#6366F1,#06B6D4); border-radius: 99px; }
.group .txt { color: #475569; font-size: .93rem; }
.note { background: #F1F5F9; border-radius: 12px; padding: .8rem 1rem; color: #475569; font-size: .92rem;
  line-height: 1.5; margin-bottom: .8rem; }
.evidence { background: #fff; border-left: 4px solid #6366F1; border-radius: 10px; padding: .9rem 1.1rem;
  color: #334155; font-size: .95rem; line-height: 1.55; border-top: 1px solid #E2E8F0;
  border-right: 1px solid #E2E8F0; border-bottom: 1px solid #E2E8F0; }
.model { background:#fff; border:1px solid #E2E8F0; border-radius:14px; padding:1rem 1.1rem; height:100%; }
.model .m { font-size:.72rem; font-weight:700; color:#4338CA; background:#EEF2FF; padding:.15rem .5rem; border-radius:6px; }
.model .nm { font-weight:700; color:#0F172A; margin-bottom:.4rem; }

/* Tabs */
.stTabs [data-baseweb="tab-list"] { gap: .3rem; border-bottom: 1px solid #E2E8F0; }
.stTabs [data-baseweb="tab"] { border-radius: 10px 10px 0 0; font-weight: 600; padding: .5rem .9rem; }
.stExpander { border-radius: 14px !important; border-color: #E2E8F0 !important; background: #fff; }

@media (max-width: 640px) {
  .hero h1 { font-size: 2rem; }
  .answer .row { flex-direction: column-reverse; align-items: flex-start; }
  .main-answer { font-size: 1.25rem; }
}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


def H(markup: str) -> None:
    """Render HTML. Leading whitespace is stripped so Markdown never treats it as a code block."""
    st.markdown("\n".join(line.strip() for line in markup.strip().splitlines()), unsafe_allow_html=True)


def esc(x) -> str:
    return html.escape(str(x))


def short(text: str, n: int = 220) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= n else text[: n - 1].rstrip() + "…"


# ───────────────────────────── Settings & state ─────────────────────────────
ALLOW_SERVER_KEYS = os.getenv("ALLOW_SERVER_KEYS", "false").lower() == "true"
MAX_QUESTION_CHARS = int(os.getenv("MAX_QUESTION_CHARS", "300"))
MAX_SAMPLES = int(os.getenv("MAX_SAMPLES", "12"))

PRESETS = [
    "What is the capital of Germany?",
    "What is the capital of Australia?",
    "Who discovered penicillin?",
    "Who won the 2030 FIFA World Cup?",
]

st.session_state.setdefault("result", None)
st.session_state.setdefault("q", "")
st.session_state.setdefault("pending", False)


def _server_key(name: str) -> str:
    return os.getenv(name, "") if ALLOW_SERVER_KEYS else ""


def _pick_preset(text: str) -> None:
    st.session_state.q = text
    st.session_state.pending = True


# ───────────────────────────── Hero ─────────────────────────────
H("""
<div class="hero">
<span class="pill">✨ Semantic Entropy · Nature 2024 method</span>
<h1>Can you trust that <span>AI answer?</span></h1>
<p>Ask any question. We ask the AI several times, compare the answers, fact-check them, and tell you in plain English whether to believe it.</p>
</div>
""")

with st.form("ask", border=False):
    question_in = st.text_input(
        "Your question",
        key="q",
        placeholder="Type any question…",
        max_chars=MAX_QUESTION_CHARS,
    )
    go = st.form_submit_button("Check this answer  →", use_container_width=True)

st.caption("Or try one of these:")
chip_cols = st.columns(len(PRESETS))
for col, p in zip(chip_cols, PRESETS):
    col.button(p, key=f"chip_{p}", on_click=_pick_preset, args=(p,), use_container_width=True)

# Advanced settings are tucked away so the home screen stays clean.
with st.expander("⚙️  Settings (AI model, number of samples)"):
    gen_mode = st.selectbox(
        "Which AI should answer?",
        ["Simulation (free, offline demo)", "ChatGPT (OpenAI API)", "Google Gemini API",
         "Kimi AI (Moonshot)", "Hugging Face (Local Models)"],
        help="Simulation needs no API key and is a good way to try the app.",
    )
    mode_key, api_key_input, model_name_input = "simulation", None, None
    if gen_mode.startswith("ChatGPT"):
        mode_key = "openai"
        api_key_input = st.text_input("OpenAI API key", type="password",
                                      placeholder="Using server key" if _server_key("OPENAI_API_KEY") else "Paste your own key"
                                      ) or _server_key("OPENAI_API_KEY") or None
        model_name_input = st.selectbox("Model", ["gpt-4o-mini", "gpt-3.5-turbo", "gpt-4o"])
    elif gen_mode.startswith("Google"):
        mode_key = "gemini"
        api_key_input = st.text_input("Gemini API key", type="password",
                                      placeholder="Using server key" if _server_key("GEMINI_API_KEY") else "Paste your own key"
                                      ) or _server_key("GEMINI_API_KEY") or None
        model_name_input = st.selectbox("Model", ["gemini-flash-latest", "gemini-2.5-flash", "gemini-2.5-flash-lite"])
    elif gen_mode.startswith("Kimi"):
        mode_key = "kimi"
        api_key_input = st.text_input("Kimi API key", type="password",
                                      placeholder="Using server key" if _server_key("KIMI_API_KEY") else "Paste your own key"
                                      ) or _server_key("KIMI_API_KEY") or None
        model_name_input = st.selectbox("Model", ["moonshot-v1-8k", "moonshot-v1-32k"])
    elif gen_mode.startswith("Hugging"):
        mode_key = "huggingface"
        model_name_input = st.text_input("Hugging Face model ID", value="HuggingFaceTB/SmolLM-135M-Instruct")

    c1, c2, c3 = st.columns(3)
    n_samples = c1.slider("Answers to collect", 3, MAX_SAMPLES, 8,
                          help="The AI is asked the same question this many times.")
    sim_threshold = c2.slider("Match strictness", 0.50, 0.95, 0.75, 0.05,
                              help="Higher = two answers must be very similar to count as the same meaning.")
    temperature = c3.slider("Creativity", 0.2, 1.5, 0.8, 0.1,
                            help="Higher = more varied answers (this is what exposes uncertainty).")
    compare_models = st.checkbox("Also compare ChatGPT, Gemini and Kimi (3 extra API calls)",
                                 disabled=mode_key == "simulation")
    custom_text = st.text_area("Advanced: paste your own AI answers (one per line)",
                               placeholder="Leave empty to generate automatically.", height=90)

# ───────────────────────────── Run ─────────────────────────────
should_run = go or st.session_state.pending
st.session_state.pending = False

if should_run:
    q = (st.session_state.q or "").strip()
    if not q:
        st.warning("Please type a question first.")
    elif len(q) > MAX_QUESTION_CHARS:
        st.warning(f"That question is too long (max {MAX_QUESTION_CHARS} characters).")
    elif mode_key in ("openai", "gemini", "kimi") and not api_key_input:
        st.warning("Add your API key under ⚙️ Settings, or switch to the free Simulation mode.")
    else:
        with st.spinner("Asking the AI several times and checking the facts…"):
            pipeline = SemanticEntropyPipeline(
                generator_mode=mode_key,
                model_name=model_name_input,
                api_key=api_key_input,
                similarity_threshold=sim_threshold,
            )
            custom_list = [l.strip() for l in custom_text.splitlines() if l.strip()] if custom_text else None
            try:
                st.session_state.result = pipeline.run(
                    question=q,
                    n_samples=n_samples,
                    temperature=temperature,
                    custom_samples=(custom_list[:MAX_SAMPLES] if custom_list else None),
                    compare_models=compare_models and mode_key != "simulation",
                )
            except Exception as e:  # keep the UI friendly if a provider fails
                st.session_state.result = None
                st.error(f"Something went wrong while running the check: {e}")

res = st.session_state.result

if not res:
    H("""
    <div class="grid" style="margin-top:1.6rem">
    <div class="card"><div class="ico">🎲</div><div class="t">1 · Ask many times</div>
    <div class="d">The same question goes to the AI several times. A confident AI repeats itself.</div></div>
    <div class="card"><div class="ico">🧩</div><div class="t">2 · Compare meanings</div>
    <div class="d">Answers are grouped by what they mean, not by the exact words used.</div></div>
    <div class="card"><div class="ico">🛡️</div><div class="t">3 · Fact-check</div>
    <div class="d">The most common answer is checked against Wikipedia and built-in facts.</div></div>
    </div>
    """)
    st.stop()

# ───────────────────────────── Results ─────────────────────────────
ov, ent, clust, ver = res["overall_status"], res["entropy"], res["clustering"], res["verification"]
conf = float(ent["confidence_score"])
n_total = res["n_samples"]
n_groups = len(clust["clusters"])
top_share = max(clust["cluster_probabilities"].values()) * 100 if clust["cluster_probabilities"] else 100
ver_bad = bool(ver["is_hallucination"])
ent_bad = bool(ent["is_hallucination_suspect"])

if not ver_bad and not ent_bad:
    tone, tag, ring = "good", "✓ Looks trustworthy", "#10B981"
    why = (f"The AI gave the same answer in {top_share:.0f}% of its {n_total} tries, and the fact-check agrees. "
           "You can rely on this answer, but still verify anything critical.")
elif ver_bad and ent_bad:
    tone, tag, ring = "bad", "⚠ Likely wrong or made up", "#EF4444"
    why = ("The AI kept changing its answer, and the fact-check could not back it up. "
           "Treat this as a possible hallucination and do not rely on it.")
elif ver_bad:
    tone, tag, ring = "bad", "⚠ Consistent, but contradicted by facts", "#EF4444"
    why = ("The AI repeated the same answer every time, but it does not match the reference we checked. "
           "Being consistent does not guarantee being right here.")
else:
    tone, tag, ring = "warn", "◔ Be careful", "#F59E0B"
    why = (f"The AI gave {n_groups} different answers across {n_total} tries, so it is unsure. "
           "The fact-check did not directly contradict the top answer, but double-check it.")

for w in res.get("generator_warnings", []):
    st.warning(w)

correction = ""
if ver_bad and ver.get("ground_truth_answer") and ver["ground_truth_answer"] != ov["answer"]:
    correction = f'<div class="correct"><b>What the reference says:</b> {esc(short(ver["ground_truth_answer"], 280))}</div>'

H(f"""
<div class="answer {tone}">
<span class="tag">{tag}</span>
<div class="qline">You asked: {esc(res["question"])}</div>
<div class="row">
<div class="body">
<div class="main-answer">{esc(short(ov["answer"], 320))}</div>
<div class="why">{why}</div>
</div>
<div class="ring" style="--p:{conf:.0f}; --c:{ring}"><div><span><b>{conf:.0f}%</b><small>CONFIDENCE</small></span></div></div>
</div>
{correction}
</div>
""")

# Key numbers in plain words
level_words = {"Low": "Very consistent", "Moderate": "Somewhat mixed", "High": "Very scattered"}
agree_txt = f"{top_share:.0f}%"
H(f"""
<div class="grid">
<div class="card stat"><div class="l">Agreement</div><div class="v">{agree_txt}</div>
<div class="d">of the {n_total} AI answers said the same thing.</div></div>
<div class="card stat"><div class="l">Different answers</div><div class="v">{n_groups}</div>
<div class="d">{"All answers meant the same." if n_groups == 1 else "Distinct meanings found among the answers."}</div></div>
<div class="card stat"><div class="l">AI consistency</div><div class="v" style="font-size:1.25rem;padding:.35rem 0">{esc(level_words.get(ent["uncertainty_level"], ent["uncertainty_level"]))}</div>
<div class="d">Uncertainty score {ent["normalized_entropy"]:.2f} (0 = sure, 1 = guessing).</div></div>
<div class="card stat"><div class="l">Fact-check</div><div class="v" style="font-size:1.25rem;padding:.35rem 0">{"Passed" if not ver_bad else "Failed"}</div>
<div class="d">{esc(ver["explanation"])}</div></div>
</div>
""")

# How we got here
H('<div class="sec">How we reached this</div><div class="sec-sub">The check ran in four simple stages.</div>')
src = ver["verification_source"]
H(f"""
<div class="step"><div class="n">1</div><div><div class="t">Asked the AI {n_total} times</div>
<div class="d">Same question, slightly different wording each time, to see if the AI stays consistent.</div></div></div>
<div class="step"><div class="n">2</div><div><div class="t">Grouped answers by meaning</div>
<div class="d">{n_total} answers collapsed into {n_groups} {"group" if n_groups == 1 else "groups"}. Different words with the same meaning count as one.</div></div></div>
<div class="step"><div class="n">3</div><div><div class="t">Measured uncertainty</div>
<div class="d">More groups with a similar size means the AI is guessing. One big group means it is sure. Result: {conf:.0f}% confidence.</div></div></div>
<div class="step"><div class="n">4</div><div><div class="t">Fact-checked the top answer</div>
<div class="d">Compared against: {esc(src)}. Match with reference: {ver["evidence_similarity"] * 100:.0f}%.</div></div></div>
""")

# Details
H('<div class="sec">Explore the details</div><div class="sec-sub">Each tab explains what you are looking at.</div>')
tabs = ["🧩 Answer groups", "🛡️ Fact-check", "🗺️ Similarity map", "🧮 Score breakdown", "📝 All answers"]
if res.get("all_models"):
    tabs.append("🤖 Model comparison")
tab_objs = st.tabs(tabs)

with tab_objs[0]:
    H(f'<div class="note">The AI’s {n_total} answers were sorted into groups that mean the same thing. '
      'A single large group means the AI is confident. Many small groups mean it is unsure.</div>')
    for cid, idxs in clust["clusters"].items():
        prob = clust["cluster_probabilities"][cid]
        rep = clust["cluster_representatives"][cid]
        label = "Most common answer" if cid == 0 else f"Alternative answer {cid}"
        H(f"""
        <div class="group"><div class="top"><span class="name">{label}</span>
        <span class="pct">{prob*100:.0f}% · {len(idxs)} of {n_total}</span></div>
        <div class="bar"><div style="width:{prob*100:.1f}%"></div></div>
        <div class="txt">“{esc(short(rep, 300))}”</div></div>
        """)

with tab_objs[1]:
    H('<div class="note">We took the most common answer and compared it with a trusted reference '
      '(a built-in fact rule or a Wikipedia article). A high match means the reference supports the answer.</div>')
    status_ok = not ver_bad
    H(f'<div class="card" style="margin-bottom:.8rem"><span class="tag" style="background:{"#D1FAE5" if status_ok else "#FEE2E2"};'
      f'color:{"#047857" if status_ok else "#B91C1C"}">{esc(ver["verification_status"])}</span>'
      f'<div class="d" style="margin-top:.6rem">{esc(ver["explanation"])}</div>'
      f'<div class="d" style="margin-top:.4rem"><b>Match with reference:</b> {ver["evidence_similarity"] * 100:.0f}% &nbsp;·&nbsp; '
      f'<b>Checked using:</b> {esc(src)}</div></div>')
    st.markdown(f"**Reference: {ver['evidence_title']}**")
    H(f'<div class="evidence">{esc(short(ver["evidence_extract"], 700))}</div>')
    if str(ver.get("evidence_url", "")).startswith("http"):
        st.link_button("Open source ↗", ver["evidence_url"])

with tab_objs[2]:
    H('<div class="note"><b>Left:</b> each dot is one AI answer. Dots close together mean similar meanings, '
      'and colors show the groups. <b>Right:</b> a heat grid where bright squares mean two answers are nearly identical.</div>')
    coords = clust["coords_2d"]
    df_plot = pd.DataFrame({
        "x": coords[:, 0], "y": coords[:, 1],
        "Group": [f"Group {l + 1}" for l in clust["cluster_labels"]],
        "Answer": [short(r, 90) for r in res["responses"]],
    })
    fig = px.scatter(df_plot, x="x", y="y", color="Group", hover_data=["Answer"], template="plotly_white",
                     color_discrete_sequence=["#6366F1", "#06B6D4", "#F59E0B", "#EF4444", "#10B981", "#A855F7"])
    fig.update_traces(marker=dict(size=15, line=dict(width=2, color="white")))
    fig.update_layout(height=340, margin=dict(l=10, r=10, t=10, b=10), xaxis_title="", yaxis_title="",
                      xaxis=dict(showticklabels=False), yaxis=dict(showticklabels=False),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#F8FAFC", legend_title_text="")
    st.plotly_chart(fig, use_container_width=True)
    names = [f"#{i + 1}" for i in range(len(res["responses"]))]
    fig2 = px.imshow(clust["similarity_matrix"], x=names, y=names, zmin=0, zmax=1, template="plotly_white",
                     color_continuous_scale=["#EEF2FF", "#6366F1", "#312E81"],
                     labels=dict(color="Similarity"))
    fig2.update_layout(height=380, margin=dict(l=10, r=10, t=10, b=10), paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig2, use_container_width=True)

with tab_objs[3]:
    H('<div class="note">Entropy is a measure of surprise. If the AI always says the same thing the score is 0 (certain). '
      'The more it spreads its answers across different meanings, the closer to 1 (guessing). '
      'Confidence is simply 100% minus that score.</div>')
    rows = []
    for cid, p in clust["cluster_probabilities"].items():
        rows.append({
            "Group": "Most common answer" if cid == 0 else f"Alternative {cid}",
            "Answers": len(clust["clusters"][cid]),
            "Share": f"{p * 100:.0f}%",
            "Uncertainty it adds": round(-(p * np.log(p)) if p > 0 else 0, 3),
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    c1, c2, c3 = st.columns(3)
    c1.metric("Raw uncertainty", ent["raw_entropy"], help="Shannon entropy of the answer groups.")
    c2.metric("Worst case", ent["max_possible_entropy"], help="Highest possible score for this many answers.")
    c3.metric("Confidence", f"{conf:.1f}%", help="(1 − normalized entropy) × 100")
    with st.expander("Show the math"):
        st.latex(r"SE = -\sum_{k=1}^{K} P(C_k)\ln P(C_k)")
        st.latex(r"SE_{norm} = \frac{SE}{\ln N}, \quad \text{Confidence} = (1 - SE_{norm}) \times 100\%")

with tab_objs[4]:
    H('<div class="note">Every raw answer the AI produced, in order. Open the group tab to see how they were merged.</div>')
    group_of = {i: g for g, idxs in clust["clusters"].items() for i in idxs}
    for i, r in enumerate(res["responses"]):
        g = group_of.get(i, 0)
        H(f'<div class="step"><div class="n">{i + 1}</div><div><div class="t">{esc(r)}</div>'
          f'<div class="d">Group {g + 1}</div></div></div>')

if res.get("all_models"):
    with tab_objs[5]:
        H('<div class="note">The same question sent to each AI provider. Do they agree?</div>')
        models = list(res["all_models"].items())
        cols = st.columns(len(models))
        for col, (key, d) in zip(cols, models):
            with col:
                H(f'<div class="model"><div class="nm">{esc(key.title())} <span class="m">{esc(d.get("model", ""))}</span></div>'
                  f'<div class="d">“{esc(short(d.get("dominant", "No response"), 260))}”</div></div>')
