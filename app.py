# streamlit_app.py
import os
import streamlit as st
from dotenv import load_dotenv
from revisionai_notion import NotionPageLoader
from revisionai_rag import RevisionRAG
from revision_scheduler import check_due_revisions, mark_page_revised

load_dotenv()

notion_token = os.getenv("NOTION_TOKEN")
groq_api_key = os.getenv("GROQ_API_KEY")

st.set_page_config(page_title="Revision AI", layout="wide")
st.title("🧠 Revision AI Dashboard")

st.subheader("🔔 Revision Reminders")
check_due_revisions()

st.divider()

st.subheader("📄 Select a Notion Page to Revise")
loader = NotionPageLoader(notion_token)
pages = loader.get_all_page_contents()

page_titles = [p["title"] for p in pages]
selected_title = st.selectbox("Choose a page:", page_titles)
selected_page = next(p for p in pages if p["title"] == selected_title)

if st.button("📚 Build RAG from Selected Page"):
    rag = RevisionRAG(groq_api_key)
    rag.build_rag_from_pages([selected_page])
    st.session_state["rag"] = rag
    st.session_state["selected_page"] = selected_page
    st.success(f"RAG built from {selected_title}")

if "rag" in st.session_state:
    rag = st.session_state["rag"]
    selected_page = st.session_state["selected_page"]

    st.subheader("💬 Ask Questions or Generate Quiz")
    mode = st.radio("Choose Mode", ["Ask a Question", "Generate Quiz"])

    if mode == "Ask a Question":
        question = st.text_input("Type your question:")
        if st.button("Get Answer") and question:
            answer = rag.ask(question)
            st.markdown(f"**💡 Answer:** {answer}")
            mark_page_revised(selected_page["title"])

    elif mode == "Generate Quiz":
        if st.button("Generate Quiz"):
            questions = rag.generate_revision_questions(selected_page["content"])
            st.text_area("📝 Quiz Questions", questions, height=300)
            mark_page_revised(selected_page["title"])
