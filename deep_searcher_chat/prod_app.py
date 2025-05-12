import os
import sys
import logging
import uuid
import time
import streamlit as st
from azure.storage.blob import BlobServiceClient
from azure.identity import DefaultAzureCredential
from io import BytesIO

# Import the improved generate_docx from helper_md_to_docx.py
from helper_md_to_docx import generate_docx

# ======================== Page Configuration ======================== #
st.set_page_config(
    page_title="VCC DeepSearcher",
    page_icon=":material/cognition:",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ======================== Logging ======================== #
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ======================== Path Setup ======================== #
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)

# ======================== Internal Imports ======================== #
from prod_cookie_manager import (
    set_browser_cookie_if_missing,
    listen_for_cookie_from_js,
    get_session_id,
)
from prod_chat_handler import (
    process_query_stream,
    revectorize_private_data,
    list_collections,
)

# ======================== Azure Configuration from Environment Variables ======================== #
STORAGE_ACCOUNT_URL = os.environ.get("AZURE_NORMAL_ACCOUNT_URL")
CONTAINER_NAME = "sandbox"  # Fixed container name as specified

if not STORAGE_ACCOUNT_URL:
    st.error("Azure Storage Account URL is not set. Please check your environment variables (AZURE_NORMAL_ACCOUNT_URL).")
    st.stop()

# ======================== Session Initialization ======================== #
set_browser_cookie_if_missing()
listen_for_cookie_from_js()
if "client_cookie" not in st.session_state:
    st.warning("Acquiring session cookie…")
    st.stop()
user_id = get_session_id()
if not user_id:
    st.error("Unable to get a valid session ID. Please refresh the page.")
    st.stop()

# Multi-user state management
if "all_users" not in st.session_state:
    st.session_state["all_users"] = {}
if user_id not in st.session_state["all_users"]:
    st.session_state["all_users"][user_id] = {
        "history": [],
        "current": {},
        "deep_search": False,
        "web_search": False,
        "thinking_messages": [],
        "collections": [],
        "form_counter": 0,
        "upload_expanded": False,  # Flag to control expander state
    }

def user_session_data():
    return st.session_state["all_users"][user_id]

MAX_HISTORY = 20

# ======================== Helpers ======================== #
@st.cache_data(ttl=60)  # Cache for 1 minute
def get_milvus_collections():
    try:
        return list_collections()
    except Exception as e:
        logger.error("Failed to fetch collections: %s", e)
        st.error("Failed to fetch collections. Please try again later.")
        return []

def history_context():
    ctx = ""
    for h in user_session_data()["history"][-5:]:
        ctx += f"User: {h['user']}\nAI: {h['ai']}\n"
    return ctx

def upload_files_to_blob(uploaded_files, user_id):
    try:
        credential = DefaultAzureCredential()
        blob_service_client = BlobServiceClient(account_url=STORAGE_ACCOUNT_URL, credential=credential)
        container_client = blob_service_client.get_container_client(CONTAINER_NAME)
        
        for file in uploaded_files:
            blob_name = f"{user_id}/{uuid.uuid4()}_{file.name}"
            blob_client = container_client.get_blob_client(blob_name)
            blob_client.upload_blob(file, overwrite=True)
        
        return True
    except Exception as e:
        logger.error(f"Upload failed: {str(e)}")
        return False

# ======================== Custom CSS ======================== #
st.markdown(
    """
    <style>
    .stApp { 
        background-color: #f9f9f9; 
        font-family: 'Arial', sans-serif; 
        padding: 20px;
        font-size: 16px;
    }
    .stMarkdown p {
        font-size: 16px;
        line-height: 1.6;
    }
    .stMarkdown h1 {
        font-size: 24px;
        margin-bottom: 16px;
    }
    .stMarkdown h2 {
        font-size: 20px;
        margin-bottom: 12px;
    }
    .stMarkdown h3 {
        font-size: 18px;
        margin-bottom: 8px;
    }
    .stMarkdown ul, .stMarkdown ol {
        padding-left: 24px;
        margin-bottom: 16px;
    }
    .stMarkdown li {
        margin-bottom: 8px;
    }
    .stMarkdown code {
        background-color: #f0f0f0;
        padding: 2px 4px;
        border-radius: 4px;
    }
    .stMarkdown pre {
        background-color: #f0f0f0;
        padding: 16px;
        border-radius: 8px;
        overflow-x: auto;
    }
    .stMarkdown table {
        width: 100%;
        border-collapse: collapse;
        margin-bottom: 16px;
    }
    .stMarkdown th, .stMarkdown td {
        border: 1px solid #ddd;
        padding: 8px;
        text-align: left;
    }
    .stMarkdown th {
        background-color: #f5f5f5;
    }
    .reference-source h4 {
        font-size: 18px;
        margin-bottom: 8px;
        color: #333;
    }
    .reference-source p {
        font-size: 16px;
        line-height: 1.6;
        color: #555;
    }
    .stButton>button {
        color: #262730;
        background-color: #f0f2f6;
        border: 1px solid #d1d5db;
        border-radius: 5px;
        padding: 10px 20px;
        font-size: 14px;
        transition: all 0.3s ease;
        width: 100%;
        margin-bottom: 10px;
    }
    .stButton>button:hover {
        background-color: #e5e7eb;
        transform: translateY(-2px);
    }
    .stTextArea textarea { 
        border-radius: 8px; 
        padding: 12px; 
        height: 200px;
        background-color: #ffffff; 
        border: 1.5px solid #e0e0e0; 
        font-size: 16px;
    }
    .stTextArea textarea:focus { 
        border-color: #007bff; 
        box-shadow: 0 0 5px rgba(0,123,255,0.3);
    }
    .stExpander { 
        border: 1px solid #e0e0e0; 
        border-radius: 5px; 
        background-color: #ffffff; 
        padding: 10px;
    }
    .thinking { 
        font-style: italic; 
        color: #555; 
        margin-top: 5px; 
        font-size: 14px;
    }
    .timer { 
        font-size: 12px; 
        color: #666666; 
        margin-top: 5px;
    }
    .message { 
        padding: 8px 12px; 
        border-radius: 4px; 
        margin-bottom: 10px;
    }
    .success { 
        background-color: #d4edda; 
        color: #155724;
    }
    .error { 
        background-color: #f8d7da; 
        color: #721c24;
    }
    /* Tab header fonts set to match DeepSearcher Config (20px) */
    .stTabs [role="tablist"] button {
        font-size: 20px !important;
        padding: 10px 20px;
        font-family: 'Arial', sans-serif;
    }
    /* Align tabs with DeepSearcher Config */
    .stTabs {
        margin-top: -40px;
    }
    /* Custom CSS for Sidebar Expander */
    .stExpander {
        margin-bottom: 10px;
    }
    .stExpander > div > div > button {
        font-size: 14px;
        color: #262730;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ======================== Sidebar ======================== #
st.sidebar.title("DeepSearcher Config")

st.sidebar.markdown(
    "<style>div.row-widget.stButton>button{white-space:nowrap;width:100%;}</style>",
    unsafe_allow_html=True
)

# Revectorize Data
revec_placeholder = st.sidebar.empty()
if st.sidebar.button("Revectorize Data", key=f"revec_{user_id}"):
    with st.spinner("Revectorizing…"):
        ok, msg = revectorize_private_data()
    if ok:
        revec_placeholder.success("Revectorization completed!")
        time.sleep(0.5)
        revec_placeholder.empty()
    else:
        if "already in progress" in msg.lower():
            revec_placeholder.warning("Revectorization already in progress.")
        else:
            revec_placeholder.error(f"Revectorization failed: {msg}")
        time.sleep(0.5)
        revec_placeholder.empty()

# Research Toggle
research_label = "Research: On" if user_session_data()["deep_search"] else "Research: Off"
if st.sidebar.button(research_label, key=f"btn_research_{user_id}"):
    user_session_data()["deep_search"] = not user_session_data()["deep_search"]
    state = "enabled" if user_session_data()["deep_search"] else "disabled"
    st.toast(f"Research mode {state}.")
    st.rerun()

# Web Toggle
web_label = "Web: On" if user_session_data()["web_search"] else "Web: Off"
if st.sidebar.button(web_label, key=f"btn_web_{user_id}"):
    user_session_data()["web_search"] = not user_session_data()["web_search"]
    state = "enabled" if user_session_data()["web_search"] else "disabled"
    st.toast(f"Web search {state}.")
    st.rerun()

# Collections Multiselect (without form)
collections_options = get_milvus_collections()
temp_selected = st.sidebar.multiselect(
    "Collections",
    options=collections_options,
    default=user_session_data()["collections"],
    key=f"col_select_{user_id}",
    label_visibility="collapsed",
    placeholder="Choose Collections",
    on_change=lambda: user_session_data().update({"collections": st.session_state[f"col_select_{user_id}"]})
)

# File Upload Widget inside Expander
upload_expanded = user_session_data().get("upload_expanded", False)
with st.sidebar.expander("Sandbox Upload & Revectorize", expanded=upload_expanded):
    with st.form(key=f"upload_form_{user_id}", clear_on_submit=True):
        uploaded_files = st.file_uploader(
            "Upload Sandbox Files",
            accept_multiple_files=True,
            label_visibility="hidden",
        )
        submit_button = st.form_submit_button("Upload Files")

        if submit_button and uploaded_files:
            with st.spinner("Uploading files to sandbox..."):
                success = upload_files_to_blob(uploaded_files, user_id)
            if success:
                st.toast("Files uploaded successfully! Consider revectorizing data.", icon="✅")
                user_session_data()["upload_expanded"] = False
                st.rerun()
            else:
                st.toast("Upload failed. Please try again.", icon="❌")
                user_session_data()["upload_expanded"] = True

# Conditional Download Buttons in Sidebar
current_convo = user_session_data()["current"]
if current_convo.get("ai"):
    user_input = current_convo['user']
    ai_response = current_convo['ai']
    
    # Markdown download
    current_md = f"**You:** {user_input}\n\n**AI:**\n{ai_response}"
    st.sidebar.download_button(
        "Download Current (.md)",
        current_md,
        file_name="current_conversation.md",
        mime="text/markdown",
        key=f"dl_current_md_{user_id}"
    )
    
    # DOCX download
    docx_file = generate_docx(user_input, ai_response)
    st.sidebar.download_button(
        "Download Current (.docx)",
        docx_file,
        file_name="current_conversation.docx",
        mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        key=f"dl_current_docx_{user_id}"
    )

# ======================== Main Area ======================== #
tab_chat, tab_reference, tab_past = st.tabs(["Current Chat", "Reference Source", "Past Conversations"])

with tab_chat:
    cur = user_session_data()["current"]
    if cur.get("user"):
        st.markdown(f"**You:** {cur['user']}")
        if cur.get("ai"):
            st.markdown(cur["ai"])

with tab_reference:
    docs = current_convo.get("retrieved", [])
    if docs:
        for i, doc in enumerate(docs, start=1):
            with st.expander(f"Reference {i}: {doc['reference']}", expanded=False):
                if doc['reference'].startswith("http://") or doc['reference'].startswith("https://"):
                    st.markdown(f"**Source:** <a href='{doc['reference']}' target='_blank' rel='noopener noreferrer'>{doc['reference']}</a>", unsafe_allow_html=True)
                else:
                    st.markdown(f"**Source:** {doc['reference']}")
                st.markdown(f"**Excerpt:**")
                st.markdown(f'<p class="reference-source">{doc["excerpt"]}</p>', unsafe_allow_html=True)
                st.markdown("---")
    else:
        st.info("No reference documents to display for the current conversation.")

with tab_past:
    if user_session_data()["history"]:
        for i, convo in enumerate(user_session_data()["history"], start=1):
            with st.expander(f"Conversation {i}: {convo['user'][:50]}...", expanded=False):
                st.markdown(f"**You:** {convo['user']}")
                st.markdown(convo["ai"])
                st.markdown("---")
    else:
        st.info("No past conversations to display.")

# ======================== Input Form ======================== #
form_key = f"input_form_{user_session_data()['form_counter']}"
st.divider()
with st.form(key=form_key, clear_on_submit=True):
    user_input = st.text_area(
        "Ask a Question",
        placeholder="Inquiry here, you may sharpen your scope via Collections first \U0001F600",
        height=200,
        label_visibility="collapsed",
        key=f"text_area_{user_session_data()['form_counter']}"
    )
    submitted = st.form_submit_button("Submit")

if submitted and user_input:
    selected_collections = st.session_state[f"col_select_{user_id}"]
    if not selected_collections:
        st.toast("No Collections Selected")
    else:
        st.toast(f"Collections Selected: {', '.join(selected_collections)}")

    if user_session_data()["current"].get("ai"):
        user_session_data()["history"].append(user_session_data()["current"])
        if len(user_session_data()["history"]) > MAX_HISTORY:
            user_session_data()["history"] = user_session_data()["history"][-MAX_HISTORY:]
    user_session_data()["current"] = {"user": user_input, "ai": None, "retrieved": None}
    user_session_data()["thinking_messages"] = []

    with st.status("Processing your question...", expanded=True) as status:
        thinking_placeholder = st.empty()
        timer_placeholder = st.empty()
        start_time = time.time()

        def update_think(message):
            current_time = time.time()
            user_session_state = user_session_data()
            user_session_state["thinking_messages"] = [
                msg for msg in user_session_state["thinking_messages"]
                if current_time - msg["timestamp"] <= 5
            ]
            user_session_state["thinking_messages"].append({
                "message": message,
                "timestamp": current_time
            })
            thinking_md = "\n".join(f'<div class="thinking">*Thinking:* {msg["message"]}</div>'
                                   for msg in user_session_state["thinking_messages"])
            thinking_placeholder.markdown(thinking_md, unsafe_allow_html=True)
            elapsed_time = int(current_time - start_time)
            timer_placeholder.markdown(f'<div class="timer">*Processing for {elapsed_time} seconds...*</div>',
                                      unsafe_allow_html=True)

        try:
            maxiter = 4 if user_session_data()["deep_search"] else 2
            final_answer, docs, _ = process_query_stream(
                user_input,
                history_context=history_context() + f"User: {user_input}\n",
                think_callback=update_think,
                maxiter=maxiter,
                collections=selected_collections,
                use_web_search=user_session_data()["web_search"]
            )
            user_session_data()["current"]["ai"] = final_answer
            user_session_data()["current"]["retrieved"] = docs
            status.update(label="Query completed!", state="complete")
        except Exception as e:
            status.update(label=f"Error: {str(e)}", state="error")

    user_session_data()["form_counter"] += 1
    st.rerun()