import os
import streamlit as st
from dotenv import load_dotenv
from revisionai_notion import NotionPageLoader
from revisionai_rag import RevisionRAG
from revision_scheduler import check_due_revisions, mark_page_revised

load_dotenv()
os.environ["TOKENIZERS_PARALLELISM"] = "false"

notion_token = os.getenv("NOTION_TOKEN")
groq_api_key = os.getenv("GROQ_API_KEY")
qdrant_url = os.getenv("QDRANT_HOST")
qdrant_api_key = os.getenv("QDRANT_API_KEY")

# Initialize RAG and Notion loader
rag = RevisionRAG(groq_api_key, qdrant_url, qdrant_api_key)
reader = NotionPageLoader(notion_token)

# Page state
if "selected_page" not in st.session_state:
    st.session_state.selected_page = None
if "selected_topic" not in st.session_state:
    st.session_state.selected_topic = "all"
if "all_pages" not in st.session_state:
    st.session_state.all_pages = reader.get_all_page_contents()

st.title("🧠 Revision AI")

# Sync option
if st.button("🔄 Sync Notion Pages"):
    reader.refresh_and_cache_pages()
    st.session_state.all_pages = reader.get_all_page_contents()
    st.success("Synced and cached latest Notion pages.")

# Show revision reminders
st.subheader("🔔 Revision Reminders")
due = check_due_revisions(display=False)
if due:
    for title, days in due:
        st.markdown(f"- **{title}**: Last revised {days} day(s) ago")
else:
    st.info("No revisions due today!")

# Option to update vectors
if st.button("🧠 Rebuild all vectorstore entries"):
    rag.build_rag_from_pages(st.session_state.all_pages)

# Topic selection
topics = rag.get_available_topics()
selected_topic = st.selectbox("📚 Select a topic", ["all"] + topics)
rag.set_topic(selected_topic)
st.session_state.selected_topic = selected_topic

# Filter pages
filtered_pages = rag.filter_pages_by_topic(st.session_state.all_pages, selected_topic)
titles = [page["title"] for page in filtered_pages]

selected_title = st.selectbox("📄 Select a Notion page", titles)
selected_page = next((p for p in filtered_pages if p["title"] == selected_title), None)

if selected_page:
    st.session_state.selected_page = selected_page
    selected_topic = rag.extract_topic_from_title(selected_page["title"])
    rag.set_topic(selected_topic)
    st.markdown(
        f"### ✅ Selected: {selected_page['title']} ({len(selected_page['content'].split())} words)"
    )

    if st.button("📥 Load Page into Vector Store"):
        rag.build_rag_from_pages([selected_page])
        st.success("Page vectors refreshed.")

    question = st.text_input("💬 Ask a question or type 'quiz'")
    if question:
        if question.strip().lower() == "quiz":
            st.subheader("📋 Revision Questions")
            questions = rag.generate_revision_questions(selected_page["content"])
            st.text_area("Generated Questions", value=questions, height=400)
            mark_page_revised(selected_page["title"])
        else:
            st.subheader("💡 Answer")
            answer = rag.ask(question)
            st.write(answer)
            mark_page_revised(selected_page["title"])

    if st.button("🔄 Select Another Page"):
        st.session_state.selected_page = None
