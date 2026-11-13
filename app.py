"""
Streamlit Web Dashboard for Semantic Entropy & Verification System.
Run with: streamlit run app.py
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import time
import html

from core.pipeline import SemanticEntropyPipeline
from core.generator import ResponseGenerator

# Set Page Config
st.set_page_config(
    page_title="Semantic Entropy & Verification Audit",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(90deg, #4F46E5, #06B6D4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        color: #94A3B8;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: rgba(30, 41, 59, 0.6);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    .metric-title {
        color: #94A3B8;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        margin-top: 0.3rem;
    }
    .status-badge-safe {
        background-color: rgba(16, 185, 129, 0.15);
        color: #10B981;
        border: 1px solid #10B981;
        padding: 0.4rem 0.8rem;
        border-radius: 8px;
        font-weight: 600;
    }
    .status-badge-warn {
        background-color: rgba(239, 68, 68, 0.15);
        color: #EF4444;
        border: 1px solid #EF4444;
        padding: 0.4rem 0.8rem;
        border-radius: 8px;
        font-weight: 600;
    }
    .cluster-box {
        background: rgba(15, 23, 42, 0.7);
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 0.8rem;
        border-left: 4px solid #4F46E5;
    }
</style>
""", unsafe_allow_html=True)

# Initialize Session State
if "pipeline_result" not in st.session_state:
    st.session_state.pipeline_result = None

# Load environment variables
import os
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Sidebar Controls
st.sidebar.image("https://img.icons8.com/fluency/96/artificial-intelligence.png", width=64)
st.sidebar.title("Configuration")

gen_mode = st.sidebar.selectbox(
    "LLM Generation Backend",
    options=[
        "ChatGPT (OpenAI API)",
        "Google Gemini API",
        "Kimi AI (Moonshot)",
        "Simulation (Offline Benchmark)",
        "Hugging Face (Local Models)"
    ],
    index=3  # Simulation by default: no API cost for anonymous visitors
)

# Server-side keys are only used if the operator explicitly opts in.
ALLOW_SERVER_KEYS = os.getenv("ALLOW_SERVER_KEYS", "false").lower() == "true"
MAX_QUESTION_CHARS = int(os.getenv("MAX_QUESTION_CHARS", "300"))
MAX_SAMPLES = int(os.getenv("MAX_SAMPLES", "12"))

def _server_key(name: str) -> str:
    return os.getenv(name, "") if ALLOW_SERVER_KEYS else ""

api_key_input = None
model_name_input = None

if gen_mode == "ChatGPT (OpenAI API)":
    mode_key = "openai"
    api_key_input = st.sidebar.text_input(
        "OpenAI API Key",
        value="",
        type="password",
        placeholder="Using server key" if _server_key("OPENAI_API_KEY") else "Paste your own key",
    ) or _server_key("OPENAI_API_KEY") or None
    model_name_input = st.sidebar.selectbox("OpenAI Model", ["gpt-4o-mini", "gpt-3.5-turbo", "gpt-4o"])
elif gen_mode == "Google Gemini API":
    mode_key = "gemini"
    api_key_input = st.sidebar.text_input(
        "Gemini API Key",
        value="",
        type="password",
        placeholder="Using server key" if _server_key("GEMINI_API_KEY") else "Paste your own key",
    ) or _server_key("GEMINI_API_KEY") or None
    model_name_input = st.sidebar.selectbox("Gemini Model", ["gemini-flash-latest", "gemini-2.5-flash", "gemini-2.5-flash-lite"])
elif gen_mode == "Kimi AI (Moonshot)":
    mode_key = "kimi"
    api_key_input = st.sidebar.text_input(
        "Kimi API Key",
        value="",
        type="password",
        placeholder="Using server key" if _server_key("KIMI_API_KEY") else "Paste your own key",
    ) or _server_key("KIMI_API_KEY") or None
    model_name_input = st.sidebar.selectbox("Kimi Model", ["moonshot-v1-8k", "moonshot-v1-32k"])
elif gen_mode == "Hugging Face (Local Models)":
    mode_key = "huggingface"
    model_name_input = st.sidebar.text_input("Hugging Face Model ID", value="HuggingFaceTB/SmolLM-135M-Instruct")
else:
    mode_key = "simulation"

st.sidebar.markdown("---")
st.sidebar.subheader("Sampling & Clustering Parameters")
n_samples = st.sidebar.slider("Number of Sampled Responses (N)", min_value=3, max_value=MAX_SAMPLES, value=8, step=1)
sim_threshold = st.sidebar.slider("Semantic Similarity Merge Threshold", min_value=0.50, max_value=0.95, value=0.75, step=0.05)
temperature = st.sidebar.slider("Sampling Temperature (τ)", min_value=0.2, max_value=1.5, value=0.8, step=0.1)

st.sidebar.markdown("---")
st.sidebar.subheader("Preset Test Benchmarks")
preset_options = [
    "Who invented the telephone?",
    "What is the capital of Germany?",
    "What is the capital of Australia?",
    "Who discovered penicillin?",
    "What was the first country to land a human on Mars?",
    "Who won the 2030 FIFA World Cup?",
    "Custom Question..."
]
selected_preset = st.sidebar.selectbox("Choose a Benchmark Scenario", preset_options, index=0)
compare_models = st.sidebar.checkbox(
    "Also compare ChatGPT, Gemini and Kimi (3 extra API calls)",
    value=False,
    disabled=mode_key == "simulation",
)

# Main Header
st.markdown('<div class="main-header">🧠 Semantic Entropy & Verification Audit</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Detecting LLM Hallucinations via Semantic Clustering & RAG Verification Layer</div>', unsafe_allow_html=True)

# Input Area
col_input, col_btn = st.columns([4, 1])
with col_input:
    if selected_preset == "Custom Question...":
        user_question = st.text_input("Enter Question for Hallucination Analysis:", value="Who invented the telephone?")
    else:
        user_question = st.text_input("Enter Question for Hallucination Analysis:", value=selected_preset)

with col_btn:
    st.write("")
    st.write("")
    run_btn = st.button("🚀 Run Analysis", type="primary", use_container_width=True)

# Custom manual responses expander
with st.expander("⚙ Optional: Provide Custom LLM Responses Manually"):
    custom_text = st.text_area(
        "Enter custom responses (one per line):",
        placeholder="Leave empty to use automatic LLM generator.\nResponse 1...\nResponse 2...",
        height=100
    )

# Execution Logic
if run_btn and not (user_question or "").strip():
    st.warning("Please enter a question.")
elif run_btn and len(user_question) > MAX_QUESTION_CHARS:
    st.warning(f"Question is too long (max {MAX_QUESTION_CHARS} characters).")
elif run_btn and mode_key in ("openai", "gemini", "kimi") and not api_key_input:
    st.warning("Enter an API key in the sidebar, or choose Simulation mode.")
elif run_btn:
    with st.spinner("Executing 7-Step Semantic Entropy & Verification Pipeline..."):
        pipeline = SemanticEntropyPipeline(
            generator_mode=mode_key,
            model_name=model_name_input,
            api_key=api_key_input,
            similarity_threshold=sim_threshold
        )
        custom_list = [line.strip() for line in custom_text.strip().split("\n") if line.strip()] if custom_text else None
        
        result = pipeline.run(
            question=user_question,
            n_samples=n_samples,
            temperature=temperature,
            custom_samples=(custom_list[:MAX_SAMPLES] if custom_list else None),
            compare_models=compare_models and mode_key != "simulation",
        )
        st.session_state.pipeline_result = result

# Display Results
res = st.session_state.pipeline_result

if not res:
    st.info("Pick a benchmark or type a question, then press **Run Analysis**.")

if res:
    for w in res.get("generator_warnings", []):
        st.warning(w)
    ov = res["overall_status"]
    ent = res["entropy"]
    clust = res["clustering"]
    ver = res["verification"]

    # Top Metric Dashboard Cards
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Confidence Score</div>
            <div class="metric-value" style="color: {'#10B981' if ent['confidence_score'] > 65 else ('#F59E0B' if ent['confidence_score'] > 40 else '#EF4444')};">
                {ent['confidence_score']}%
            </div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Normalized Entropy</div>
            <div class="metric-value" style="color: #6366F1;">
                {ent['normalized_entropy']}
            </div>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Semantic Clusters</div>
            <div class="metric-value" style="color: #38BDF8;">
                {len(clust['clusters'])} / {res['n_samples']}
            </div>
        </div>
        """, unsafe_allow_html=True)

    with c4:
        is_safe = not (ent["is_hallucination_suspect"] or ver["is_hallucination"])
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Audit Status</div>
            <div class="metric-value" style="font-size: 1.2rem; margin-top: 0.6rem; color: {'#10B981' if is_safe else '#EF4444'};">
                {'✓ Verified Factual' if is_safe else '⚠ Hallucination Risk'}
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.write("")

    # Step 7 Summary Alert Banner
    if is_safe:
        st.success(f"**Final Verdict**: {ver['verification_status']}\n\n**Dominant Answer**: {ov['answer']}\n\n**Confidence**: {ov['confidence']} | **Semantic Entropy**: {ov['semantic_entropy']}")
    else:
        st.error(f"**Final Verdict**: {ov['status']}\n\n**Proposed LLM Answer**: {ov['answer']}\n\n**Confidence**: {ov['confidence']} | **Entropy**: {ov['semantic_entropy']}\n\n**Verified Ground Truth**: {ov['verified_answer']}")

    st.write("---")

    # 🤖 ALL 3 AI RESPONSES ON SCREEN SECTION
    all_models = res.get("all_models", {})
    if all_models:
        st.markdown("### 🤖 Live Multi-Model Comparison (All 3 AIs on Screen)")

        col_gpt, col_gem, col_kimi = st.columns(3)
    
        with col_gpt:
            gpt_data = all_models.get("chatgpt", {})
            st.markdown(f"""
            <div style="background: rgba(16, 185, 129, 0.08); border: 2px solid #10B981; border-radius: 12px; padding: 1.2rem; min-height: 180px; box-shadow: 0 4px 12px rgba(16, 185, 129, 0.1);">
                <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 0.6rem;">
                    <span style="font-weight: 700; color: #10B981; font-size: 1.1rem;">🟢 ChatGPT</span>
                    <span style="background: #10B981; color: white; padding: 0.2rem 0.5rem; border-radius: 6px; font-size: 0.75rem; font-weight: 600;">{gpt_data.get('model', 'gpt-4o-mini')}</span>
                </div>
                <div style="font-size: 0.95rem; color: #F1F5F9; line-height: 1.5; margin-top: 0.5rem;">
                    "{html.escape(str(gpt_data.get('dominant', 'No response')))}"
                </div>
            </div>
            """, unsafe_allow_html=True)

        with col_gem:
            gem_data = all_models.get("gemini", {})
            st.markdown(f"""
            <div style="background: rgba(59, 130, 246, 0.08); border: 2px solid #3B82F6; border-radius: 12px; padding: 1.2rem; min-height: 180px; box-shadow: 0 4px 12px rgba(59, 130, 246, 0.1);">
                <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 0.6rem;">
                    <span style="font-weight: 700; color: #3B82F6; font-size: 1.1rem;">🔵 Google Gemini</span>
                    <span style="background: #3B82F6; color: white; padding: 0.2rem 0.5rem; border-radius: 6px; font-size: 0.75rem; font-weight: 600;">{gem_data.get('model', 'gemini-flash-latest')}</span>
                </div>
                <div style="font-size: 0.95rem; color: #F1F5F9; line-height: 1.5; margin-top: 0.5rem;">
                    "{html.escape(str(gem_data.get('dominant', 'No response')))}"
                </div>
            </div>
            """, unsafe_allow_html=True)

        with col_kimi:
            kimi_data = all_models.get("kimi", {})
            st.markdown(f"""
            <div style="background: rgba(168, 85, 247, 0.08); border: 2px solid #A855F7; border-radius: 12px; padding: 1.2rem; min-height: 180px; box-shadow: 0 4px 12px rgba(168, 85, 247, 0.1);">
                <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 0.6rem;">
                    <span style="font-weight: 700; color: #A855F7; font-size: 1.1rem;">🟣 Kimi AI</span>
                    <span style="background: #A855F7; color: white; padding: 0.2rem 0.5rem; border-radius: 6px; font-size: 0.75rem; font-weight: 600;">{kimi_data.get('model', 'moonshot-v1-8k')}</span>
                </div>
                <div style="font-size: 0.95rem; color: #F1F5F9; line-height: 1.5; margin-top: 0.5rem;">
                    "{html.escape(str(kimi_data.get('dominant', 'No response')))}"
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.write("")
    st.write("---")

    # Step-by-Step Detailed Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🔬 Step 2 & 4: Clusters & 2D Space",
        "📊 Step 3 & 4: Similarity Heatmap",
        "🧮 Step 5: Entropy Math Breakdown",
        "🛡 Step 6: RAG Verification Layer",
        "📋 Full 7-Step Trace"
    ])

    with tab1:
        st.subheader("Semantic Equivalence Clusters")
        col_clust_list, col_plot = st.columns([1.2, 1.8])

        with col_clust_list:
            st.markdown(f"**Grouped into {len(clust['clusters'])} Semantic Meaning Classes:**")
            for cid, indices in clust["clusters"].items():
                prob = clust["cluster_probabilities"][cid]
                rep = clust["cluster_representatives"][cid]
                st.markdown(f"""
                <div class="cluster-box">
                    <strong style="color: #38BDF8;">Cluster #{cid + 1}</strong> ({len(indices)} samples, <strong>{prob*100:.1f}%</strong>)<br>
                    <em>Representative:</em> "{html.escape(str(rep))}"
                </div>
                """, unsafe_allow_html=True)

        with col_plot:
            st.markdown("**2D Semantic Cluster Projection (PCA):**")
            coords = clust["coords_2d"]
            labels = [f"Cluster {l+1}" for l in clust["cluster_labels"]]
            texts = [f"#{i+1}: {r[:80]}..." for i, r in enumerate(res["responses"])]

            df_plot = pd.DataFrame({
                "PCA Dim 1": coords[:, 0],
                "PCA Dim 2": coords[:, 1],
                "Cluster": labels,
                "Response": texts
            })

            fig = px.scatter(
                df_plot,
                x="PCA Dim 1",
                y="PCA Dim 2",
                color="Cluster",
                hover_data=["Response"],
                template="plotly_dark",
                title="Semantic Distance Space",
                size_max=15
            )
            fig.update_traces(marker=dict(size=14, line=dict(width=1, color='White')))
            fig.update_layout(margin=dict(l=20, r=20, t=40, b=20), height=360)
            st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.subheader("Cosine Similarity Matrix")
        sim_matrix = clust["similarity_matrix"]
        sample_names = [f"Sample {i+1}" for i in range(len(res["responses"]))]
        
        fig_heat = px.imshow(
            sim_matrix,
            labels=dict(x="Sample", y="Sample", color="Cosine Similarity"),
            x=sample_names,
            y=sample_names,
            color_continuous_scale="Viridis",
            zmin=0.0,
            zmax=1.0,
            template="plotly_dark"
        )
        fig_heat.update_layout(height=420, margin=dict(l=20, r=20, t=20, b=20))
        st.plotly_chart(fig_heat, use_container_width=True)

    with tab3:
        st.subheader("Semantic Entropy Calculation Details")
        st.latex(r"SE = - \sum_{k=1}^{K} P(C_k) \ln P(C_k)")
        st.latex(r"SE_{norm} = \frac{SE}{\ln(N)}, \quad \text{Confidence} = (1 - SE_{norm}) \times 100\%")

        entropy_breakdown = []
        for cid, prob in clust["cluster_probabilities"].items():
            term = - (prob * np.log(prob)) if prob > 0 else 0
            entropy_breakdown.append({
                "Cluster ID": f"Cluster {cid+1}",
                "Sample Count": len(clust["clusters"][cid]),
                "Probability P(C_k)": f"{prob:.3f}",
                "Entropy Contribution -p*ln(p)": f"{term:.4f}"
            })
        st.dataframe(pd.DataFrame(entropy_breakdown), use_container_width=True)

        st.info(f"""
        - **Raw Shannon Entropy**: `{ent['raw_entropy']}`
        - **Maximum Theoretical Entropy ln({res['n_samples']})**: `{ent['max_possible_entropy']}`
        - **Normalized Entropy**: `{ent['normalized_entropy']}` ({ent['uncertainty_level']} Uncertainty)
        - **Confidence Score**: `{ent['confidence_score']}%`
        """)

    with tab4:
        st.subheader("Verification Layer: Ground Truth & Evidence Cross-Check")
        col_ver_details, col_evi_box = st.columns([1, 1])

        with col_ver_details:
            st.markdown(f"**Verification Engine:** `{ver['verification_source']}`")
            st.markdown(f"**Status:** `{ver['verification_status']}`")
            st.markdown(f"**Semantic Alignment Score:** `{ver['evidence_similarity'] * 100:.1f}%`")
            st.markdown(f"**Explanation:** {ver['explanation']}")
            if ver.get("evidence_url"):
                st.markdown(f"🔗 **Reference Link:** [{ver['evidence_title']}]({ver['evidence_url']})")

        with col_evi_box:
            st.markdown("**Retrieved Knowledge / Rule Extract:**")
            st.info(f"**{ver['evidence_title']}**\n\n\"{ver['evidence_extract']}\"")

    with tab5:
        st.subheader("Complete 7-Step Execution Trace")
        st.markdown(f"**Step 1: User Input Question**\n> {res['question']}")
        
        st.markdown("**Step 2: Sampled LLM Responses**")
        for idx, resp in enumerate(res["responses"]):
            st.markdown(f"- **Sample {idx+1}**: {resp}")

        st.markdown(f"**Step 3: Embeddings Generated**\n> Computed `{len(res['responses'])}` dense vector representations using text embedder.")
        st.markdown(f"**Step 4: Cosine Similarity & Clustering**\n> Resulted in `{len(clust['clusters'])}` distinct meaning clusters.")
        st.markdown(f"**Step 5: Semantic Entropy Calculated**\n> Raw SE = `{ent['raw_entropy']}` | Normalized SE = `{ent['normalized_entropy']}` | Confidence = `{ent['confidence_score']}%`.")
        st.markdown(f"**Step 6: Verification Applied**\n> Source: `{ver['verification_source']}` | Alignment: `{ver['evidence_similarity']*100:.1f}%`.")
        st.markdown(f"**Step 7: Final Synthesized Report**\n> Status: `{ov['status']}` | Dominant: \"{ov['answer']}\" | Verified: \"{ov['verified_answer']}\"")
