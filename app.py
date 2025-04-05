# revision_ai_app.py

import os
import streamlit as st
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from revisionai_notion import NotionPageLoader
from revisionai_rag import RevisionRAG

# Load secrets or set your keys
NOTION_TOKEN = os.getenv("NOTION_TOKEN") or st.secrets["NOTION_TOKEN"]
GROQ_API_KEY = os.getenv("GROQ_API_KEY") or st.secrets["GROQ_API_KEY"]

# Initialize classes
notion_loader = NotionPageLoader(token=NOTION_TOKEN)
rag = RevisionRAG(groq_api_key=GROQ_API_KEY)

# --- STREAMLIT UI ---
st.set_page_config(page_title="Revision AI", layout="wide")
st.title("📘 Revision AI")

st.sidebar.header("📄 Notion Pages")
pages_data = notion_loader.get_all_page_contents()
page_titles = [page["title"] for page in pages_data]

selected_titles = st.sidebar.multiselect("Select Pages to Revise", page_titles)
timestamp_filter = st.sidebar.slider(
    "Only include blocks updated in last X days:", 0, 60, 0
)

if selected_titles:
    combined_text = ""
    for page in pages_data:
        if page["title"] in selected_titles:
            blocks = notion_loader.get_page_blocks(
                page["id"], filter_last_edited_days=timestamp_filter
            )
            block_text = "\n".join([b["text"] for b in blocks])
            combined_text += f"\n# {page['title']}\n" + block_text

    st.success(f"🔍 Loaded content from {len(selected_titles)} pages.")

    if st.button("📚 Build RAG for Selected Pages"):
        rag.build_rag_from_text(combined_text)
        st.session_state["rag_ready"] = True
        st.success("✅ RAG built! You can now ask questions.")

if st.session_state.get("rag_ready"):
    st.header("❓ Ask Questions About Your Notes")
    question = st.text_input("Type your question")
    if question:
        answer = rag.ask(question)
        st.markdown(f"**💡 Answer:** {answer}")

# --- REVISION SCHEDULER ---
st.sidebar.header("🕒 Revision Reminder")
reminder_days = ["Monday", "Thursday"]
today = datetime.today().strftime("%A")
if today in reminder_days:
    st.sidebar.success("⏰ Reminder: Time to revise selected topics today!")

# --- OPTIONAL QUIZ GENERATION PLACEHOLDER ---
if st.session_state.get("rag_ready"):
    st.subheader("📝 Practice Zone (Coming Soon)")
    st.info(
        "We'll generate MCQs, one-word answers, short/long questions, and code comprehension based on your notes."
    )
