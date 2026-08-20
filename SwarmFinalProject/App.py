import base64
import os
import time
from concurrent.futures import ThreadPoolExecutor

import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from Chunking import chunk_text
from Coder import run_coder
from Doc_Helper import extract_text_from_pdf
from dotenv import load_dotenv
from Planner import run_planner
from Researcher import run_researcher
from Tester import run_tester

import streamlit as st

load_dotenv()

st.set_page_config(page_title="Swarm AI", layout="wide")

chroma_client = chromadb.PersistentClient(path="./final_project_db")
embedding_function = OpenAIEmbeddingFunction(
    api_key=os.getenv("OPENAI_API_KEY"), model_name="text-embedding-3-small"
)
collection = chroma_client.get_or_create_collection(
    name="swarm_final_project", embedding_function=embedding_function
)

# -----------------------------------------------
# ASSETS
# -----------------------------------------------


def file_to_base64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


DIDOT_B64 = file_to_base64("fonts/Didot.otf")
DIDOT_BOLD_B64 = file_to_base64("fonts/Didot Bold.otf")
DIDOT_ITALIC_B64 = file_to_base64("fonts/Didot Italic.otf")
DIDOT_TITLE_B64 = file_to_base64("fonts/Didot Title.otf")

# -----------------------------------------------
# STATE
# -----------------------------------------------

if "status" not in st.session_state:
    st.session_state.status = {
        "planner": "green",
        "research": "green",
        "coding": "green",
        "review": "green",
    }

if "messages" not in st.session_state:
    st.session_state.messages = []

if "thinking" not in st.session_state:
    st.session_state.thinking = False

COLORS = {"green": "#2ecc71", "orange": "#e67e22"}

# -----------------------------------------------
# STYLE
# -----------------------------------------------

st.markdown(
    f"""
<style>
@font-face {{
    font-family: 'Didot';
    src: url(data:font/otf;base64,{DIDOT_B64}) format('opentype');
    font-weight: normal;
}}
@font-face {{
    font-family: 'Didot';
    src: url(data:font/otf;base64,{DIDOT_BOLD_B64}) format('opentype');
    font-weight: bold;
}}
@font-face {{
    font-family: 'Didot';
    src: url(data:font/otf;base64,{DIDOT_ITALIC_B64}) format('opentype');
    font-style: italic;
}}
@font-face {{
    font-family: 'Didot Title';
    src: url(data:font/otf;base64,{DIDOT_TITLE_B64}) format('opentype');
}}

html, body, [class*="css"], input, textarea, button, p, div, span, h1, h2, h3, h4, h5 {{
    font-family: 'Didot', serif !important;
}}

/* Main Application Semi-Transparent Blue Background */
.stApp {{
    background-color: rgba(0, 0, 242, 0.7) !important;
}}

/* Sidebar Transparency Adjustment */
[data-testid="stSidebar"] {{
    background-color: rgba(255, 255, 255, 0.9) !important;
    color: #0000F2;
}}

[data-testid="stSidebar"] * {{
    color: #0000F2 !important;
}}

h1, h2, h3, h4 {{
    font-family: 'Didot Title', serif !important;
    letter-spacing: 2px;
    color: #FFFFFF !important;
}}

/* ---------- SWARM CANVAS ---------- */

.node {{
    border: 2.5px solid;
    border-radius: 14px;
    padding: 12px 14px;
    text-align: center;
    font-weight: bold;
    font-size: 15px;
    background-color: rgba(0, 0, 242, 0.6);
    box-shadow: 0 4px 12px rgba(0,0,0,0.25);
    margin: 0 auto;
    width: 120px;
    letter-spacing: 1px;
    transition: all 0.35s ease;
}}

.line {{
    text-align: center;
    color: #FFFFFF;
    line-height: 1.4;
    font-size: 14px;
}}

.row {{
    display: flex;
    justify-content: center;
    gap: 20px;
    margin: 4px 0;
}}

.complete-badge {{
    text-align: center;
    margin-top: 16px;
    font-weight: bold;
    color: #0000F2;
    background-color: rgba(255, 255, 255, 0.95);
    display: block;
    padding: 8px 14px;
    border-radius: 20px;
    font-size: 14px;
    letter-spacing: 1px;
    box-shadow: 0 3px 8px rgba(0,0,0,0.3);
}}

/* =========================================
   SWARM AI — CHAT BOX STYLES
   ========================================= */

/* ---------- ALL CHAT MESSAGES ---------- */
[class*="st-key-msg_"] {{
    border-radius: 16px !important;
    padding: 4px 10px !important;
    border: 1px solid rgba(255, 255, 255, 0.8) !important;
    color: #FFFFFF !important;
    font-family: 'Didot', serif !important;
}}

/* ---------- USER MESSAGE ---------- */
[class*="st-key-msg_"][class*="_user"] {{
    background-color: rgba(128, 128, 128, 0.35) !important;
    border-color: rgba(255, 255, 255, 0.8) !important;
    border-radius: 16px !important;
    padding: 8px 14px !important;
    color: #FFFFFF !important;
    box-shadow: 0 2px 6px rgba(0, 0, 0, 0.25) !important;
    backdrop-filter: blur(4px);
}}

/* ---------- ASSISTANT MESSAGE ---------- */
[class*="st-key-msg_"][class*="_assistant"] {{
    background-color: rgba(0, 0, 242, 0.5) !important;
    border-color: rgba(255, 255, 255, 0.8) !important;
    border-radius: 16px !important;
    padding: 8px 14px !important;
    color: #FFFFFF !important;
    box-shadow: 0 2px 6px rgba(0, 0, 0, 0.25) !important;
    backdrop-filter: blur(4px);
}}

/* ---------- CHAT INPUT / ASK BAR ---------- */
.ask-bar [data-testid="stForm"] {{
    border: 1.5px solid rgba(255, 255, 255, 0.8) !important;
    border-radius: 20px !important;
    background-color: rgba(0, 0, 242, 0.6) !important;
    padding: 10px 14px !important;
    box-shadow: 0 3px 10px rgba(0, 0, 0, 0.25) !important;
    backdrop-filter: blur(6px);
}}

/* ---------- TEXT INPUT ---------- */
.ask-bar input {{
    border: none !important;
    outline: none !important;
    background: transparent !important;
    color: #FFFFFF !important;
    font-family: 'Didot', serif !important;
    font-size: 16px !important;
}}

/* ---------- INPUT PLACEHOLDER ---------- */
.ask-bar input::placeholder {{
    color: #C9C9FF !important;
    opacity: 1 !important;
}}

/* ---------- SEND BUTTON ---------- */
.ask-bar [data-testid="stFormSubmitButton"] button {{
    border-radius: 50% !important;
    width: 38px !important;
    height: 38px !important;
    background-color: #FFFFFF !important;
    color: #0000F2 !important;
    border: none !important;
    font-family: 'Didot', serif !important;
    font-size: 18px !important;
    transition: all 0.2s ease !important;
}}

/* ---------- SEND BUTTON HOVER ---------- */
.ask-bar [data-testid="stFormSubmitButton"] button:hover {{
    background-color: #C9C9FF !important;
    transform: scale(1.05);
}}

/* ---------- SUPER SMALL UPLOADER BUTTON ---------- */
.ask-bar [data-testid="stFileUploader"] {{
    max-width: 110px !important;
    margin-top: 2px !important;
}}

.ask-bar [data-testid="stFileUploader"] section {{
    padding: 0px !important;
    background: transparent !important;
    border: none !important;
    min-height: unset !important;
}}

.ask-bar [data-testid="stFileUploader"] section button {{
    background-color: #FFFFFF !important;
    color: #0000F2 !important;
    font-size: 9px !important;
    padding: 2px 6px !important;
    border-radius: 4px !important;
    height: 18px !important;
    min-height: 18px !important;
    line-height: 1 !important;
    border: none !important;
    margin: 0 !important;
}}

.ask-bar [data-testid="stFileUploaderDropzoneInstructions"],
.ask-bar [data-testid="stFileUploader"] small,
.ask-bar [data-testid="stFileUploader"] span {{
    display: none !important;
}}

/* ---------- GENERAL CHAT TEXT ---------- */
[class*="st-key-msg_"] p {{
    color: #FFFFFF !important;
    font-family: 'Didot', serif !important;
    font-size: 15px !important;
    line-height: 1.5 !important;
}}

/* ---------- CHAT CONTAINER ---------- */
.stChatMessage {{
    border-radius: 16px !important;
    background-color: rgba(0, 0, 242, 0.5) !important;
    border: 1px solid rgba(255, 255, 255, 0.8) !important;
    backdrop-filter: blur(4px);
}}

/* ---------- GENERAL BUTTONS ---------- */
button {{
    font-family: 'Didot', serif !important;
    border-radius: 12px !important;
}}

/* ---------- COLORS ---------- */
:root {{
    --bg-main: rgba(0, 0, 242, 0.7);
    --bg-sidebar: rgba(0, 0, 242, 0.7);
    --text-light: #FFFFFF;
    --text-muted: #C9C9FF;
}}
</style>
""",
    unsafe_allow_html=True,
)

# -----------------------------------------------
# SIDEBAR — DIAGRAM
# -----------------------------------------------


def node(label, key):
    status = st.session_state.status[key]
    color = COLORS[status]
    return f'<div class="node" style="border-color:{color}; color:{color};">{label}</div>'


def render_diagram():
    return f"""
    <h4 style='text-align:center;'>SWARM CANVAS</h4>
    <div style="text-align:center; padding-bottom:14px;">
        {node("Planner", "planner")}
        <div class="line">|<br>|</div>
        <div class="line">┌──┴──┐</div>
        <div class="line">▼&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;▼</div>
        <div class="row">
            {node("Research", "research")}
            {node("Coding", "coding")}
        </div>
        <div class="line">|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|</div>
        <div class="line">└──┬──┘</div>
        <div class="line">▼</div>
        {node("Review", "review")}
    </div>
    """


with st.sidebar:
    diagram_placeholder = st.empty()
    diagram_placeholder.markdown(render_diagram(), unsafe_allow_html=True)

    complete_placeholder = st.empty()

# -----------------------------------------------
# MAIN — HEADER & CHAT
# -----------------------------------------------

st.markdown("""
    <div style="text-align: center; margin-top: 10px; margin-bottom: 30px;">
        <div style="
            color: #FFFFFF;
            font-family: 'Didot', serif;
            font-weight: bold;
            font-size: 13px;
            letter-spacing: 3.5px;
            text-transform: uppercase;
            margin-bottom: 12px;
        ">
            WORKS TOGETHER &bull; MOVES INDEPENDENTLY &bull; SHARES INFORMATION &bull; ADAPTS QUICKLY
        </div>
        <h1 style="
            color: #FFFFFF !important;
            font-family: 'Didot Title', 'Didot', serif !important;
            font-weight: bold !important;
            font-size: 68px !important;
            letter-spacing: 8px !important;
            text-transform: uppercase;
            margin: 0 !important;
            padding: 0 !important;
        ">
            SWARM AI
        </h1>
    </div>
""", unsafe_allow_html=True)

for i, message in enumerate(st.session_state.messages):
    with st.container(border=True, key=f"msg_{i}_{message['role']}"):
        st.markdown(message["content"])

if st.session_state.thinking:
    with st.container(border=True, key="msg_thinking_assistant"):
        st.markdown("*Swarming...*")

st.markdown('<div class="ask-bar">', unsafe_allow_html=True)

with st.form("ask_form", clear_on_submit=True, border=False):
    text_col, send_col = st.columns([11, 1])

    with text_col:
        prompt = st.text_input(
            "Ask anything",
            placeholder="Ask anything",
            label_visibility="collapsed",
        )

    with send_col:
        submitted = st.form_submit_button("➤")

    #uploaded_file = st.file_uploader(
    #    "Attach", type=["pdf", "txt"], label_visibility="collapsed"
    #)

st.markdown("</div>", unsafe_allow_html=True)

if submitted and prompt:
    #if uploaded_file:
     #   raw_text = (
      #      extract_text_from_pdf(uploaded_file)
       #     if uploaded_file.name.endswith(".pdf")
        #    else uploaded_file.read().decode("utf-8")
        #)
        #chunks = chunk_text(raw_text)
        #collection.upsert(
        #    documents=chunks, ids=[f"chunk_{i}" for i in range(len(chunks))]
        #)

    st.session_state.messages.append({"role": "user", "content": prompt})
    st.session_state.thinking = True
    st.rerun()

if (
    st.session_state.thinking
    and st.session_state.messages
    and st.session_state.messages[-1]["role"] == "user"
):
    prompt = st.session_state.messages[-1]["content"]

    document_context = (
        "\n\n".join(collection.get(include=["documents"])["documents"])
        if collection.count() > 0
        else ""
    )
    task = f"USER GOAL:\n{prompt}\n\nREFERENCE KNOWLEDGE:\n{document_context}"

    st.session_state.status["planner"] = "orange"
    st.session_state.status["research"] = "orange"
    st.session_state.status["coding"] = "orange"
    diagram_placeholder.markdown(render_diagram(), unsafe_allow_html=True)

    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {
            "planner": executor.submit(run_planner, task),
            "research": executor.submit(run_researcher, task),
            "coding": executor.submit(run_coder, task),
        }

        results = {}

        while futures:
            done_keys = [
                key for key, future in futures.items() if future.done()
            ]

            for key in done_keys:
                results[key] = futures[key].result()
                st.session_state.status[key] = "green"
                del futures[key]

            diagram_placeholder.markdown(
                render_diagram(), unsafe_allow_html=True
            )

            if futures:
                time.sleep(0.5)

    plan = results["planner"]
    research = results["research"]
    code = results["coding"]

    st.session_state.status["review"] = "orange"
    diagram_placeholder.markdown(render_diagram(), unsafe_allow_html=True)

    tester_task = f"""
USER GOAL:
{prompt}

PLANNER:
{plan}

RESEARCHER:
{research}

CODER:
{code}

Reply as a single natural chat message, not a formal QA report.

If the user's message was casual conversation (a greeting, small talk, a
question that didn't ask for anything to be built) and no real code or
project was produced, just reply naturally and briefly, in plain
conversational language. Do NOT include a code block. Do NOT produce a
verdict, pass/fail list, or QA-style structure.

If the user actually asked for something to be built and code was produced,
write a short conversational summary of what was built and whether it meets
the goal, then include the final code in a single code block.

Never output an empty or placeholder code block.
"""

    review = run_tester(tester_task)

    st.session_state.status["review"] = "green"
    diagram_placeholder.markdown(render_diagram(), unsafe_allow_html=True)
    complete_placeholder.markdown(
        "<div class='complete-badge'>✅ COMPLETE</div>", unsafe_allow_html=True
    )

    st.session_state.messages.append({"role": "assistant", "content": review})
    st.session_state.thinking = False

    st.rerun()