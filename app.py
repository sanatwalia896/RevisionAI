import os
import streamlit as st
from dotenv import load_dotenv
from revisionai_notion import NotionPageLoader
from revisionai_rag import RevisionRAG
from revision_scheduler import check_due_revisions, mark_page_revised

# Page configuration
st.set_page_config(
    page_title="RevisionAI - Your Knowledge Assistant", page_icon="🧠", layout="wide"
)


# Load environment variables
def initialize_environment():
    load_dotenv()
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    required_vars = ["NOTION_TOKEN", "GROQ_API_KEY", "QDRANT_HOST", "QDRANT_API_KEY"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]

    if missing_vars:
        st.error(f"Missing required environment variables: {', '.join(missing_vars)}")
        st.stop()

    return {
        "notion_token": os.getenv("NOTION_TOKEN"),
        "groq_api_key": os.getenv("GROQ_API_KEY"),
        "qdrant_url": os.getenv("QDRANT_HOST"),
        "qdrant_api_key": os.getenv("QDRANT_API_KEY"),
    }


# Initialize services
def initialize_services(env_vars):
    try:
        rag = RevisionRAG(
            env_vars["groq_api_key"], env_vars["qdrant_url"], env_vars["qdrant_api_key"]
        )
        reader = NotionPageLoader(env_vars["notion_token"])
        return rag, reader
    except Exception as e:
        st.error(f"Failed to initialize services: {str(e)}")
        st.stop()


# Load all Notion pages
def get_all_pages(reader):
    try:
        return reader.get_all_page_contents()
    except Exception as e:
        st.error(f"Error fetching Notion pages: {str(e)}")
        return []


# Initialize
env_vars = initialize_environment()
rag, reader = initialize_services(env_vars)

# Session State
if "selected_page" not in st.session_state:
    st.session_state.selected_page = None
if "selected_topic" not in st.session_state:
    st.session_state.selected_topic = "all"
if "all_pages" not in st.session_state:
    st.session_state.all_pages = get_all_pages(reader)
if "question_input" not in st.session_state:
    st.session_state.question_input = ""
if "answer_history" not in st.session_state:
    st.session_state.answer_history = []
if "show_page_content" not in st.session_state:
    st.session_state.show_page_content = False

# Title
st.title("🧠 RevisionAI")
st.caption("Your AI-powered study assistant")

# Sidebar
with st.sidebar:
    st.header("Tools & Settings")

    if st.button("🔄 Sync Notion Pages", use_container_width=True):
        with st.spinner("Syncing pages from Notion..."):
            try:
                reader.refresh_and_cache_pages()
                st.session_state.all_pages = reader.get_all_page_contents()
                st.success("✅ Synced and cached latest Notion pages.")
            except Exception as e:
                st.error(f"Sync failed: {str(e)}")

    if st.button("🧠 Rebuild Vectorstore", use_container_width=True):
        with st.spinner("Building vector embeddings..."):
            try:
                rag.build_rag_from_pages(st.session_state.all_pages)
                st.success("✅ Vector store rebuilt successfully")
            except Exception as e:
                st.error(f"Vector store update failed: {str(e)}")

    st.subheader("🔔 Revision Reminders")
    due = check_due_revisions(display=False)
    if due:
        for title, days in due:
            st.markdown(f"- **{title}**: Last revised {days} day(s) ago")
    else:
        st.info("No revisions due today!")

    st.subheader("📊 Statistics")
    st.metric("Total Pages", len(st.session_state.all_pages))

    with st.expander("ℹ️ Help"):
        st.markdown(
            """
            **How to use RevisionAI:**
            1. Select a topic and page from your Notion workspace
            2. Ask questions or generate revision quizzes
            3. Review and learn from AI-generated answers

            **Commands:**
            - Type 'quiz' to generate revision questions
            - Use clear, specific questions for best results
            """
        )

# Main layout
col1, col2 = st.columns([1, 2])

# Left Column - Page Selection
with col1:
    st.subheader("📚 Select Content")

    try:
        topics = rag.get_available_topics()
        selected_topic = st.selectbox(
            "Select a topic",
            ["all"] + topics,
            index=(
                ["all"] + topics.index(st.session_state.selected_topic)
                if st.session_state.selected_topic in topics
                else 0
            ),
        )
        rag.set_topic(selected_topic)
        st.session_state.selected_topic = selected_topic
    except Exception as e:
        st.error(f"Error loading topics: {str(e)}")
        selected_topic = "all"

    filtered_pages = rag.filter_pages_by_topic(
        st.session_state.all_pages, selected_topic
    )

    if not filtered_pages:
        st.warning("No pages found for this topic.")
    else:
        titles = [page["title"] for page in filtered_pages]
        selected_title = st.selectbox("Select a page", titles)
        selected_page = next(
            (p for p in filtered_pages if p["title"] == selected_title), None
        )

        if selected_page:
            st.session_state.selected_page = selected_page
            selected_topic = rag.extract_topic_from_title(selected_page["title"])
            rag.set_topic(selected_topic)

            st.success(f"✅ Selected: {selected_page['title']}")
            word_count = len(selected_page["content"].split())
            st.caption(f"Word count: {word_count}")

            if st.button("📥 Load Page into Vector Store", use_container_width=True):
                with st.spinner("Loading page vectors..."):
                    rag.build_rag_from_pages([selected_page])
                    st.success("✅ Page vectors refreshed")

            st.session_state.show_page_content = st.toggle(
                "Show page content", st.session_state.show_page_content
            )
            if st.session_state.show_page_content:
                with st.expander("Page content", expanded=True):
                    st.markdown(
                        selected_page["content"][:1000] + "..."
                        if len(selected_page["content"]) > 1000
                        else selected_page["content"]
                    )
                    if len(selected_page["content"]) > 1000:
                        st.caption(
                            "Content truncated. Full content used for AI responses."
                        )

# Right Column - Interaction
with col2:
    if st.session_state.selected_page:
        st.subheader("💬 Ask or Revise")

        with st.form("question_form", clear_on_submit=False):
            st.session_state.question_input = st.text_input(
                "Ask a question or type 'quiz' to generate revision questions",
                value=st.session_state.question_input,
            )
            col1, col2 = st.columns([1, 1])
            with col1:
                submit_button = st.form_submit_button(
                    "🚀 Submit", use_container_width=True
                )
            with col2:
                clear_button = st.form_submit_button(
                    "🧹 Clear History", use_container_width=True
                )
                if clear_button:
                    st.session_state.answer_history = []
                    st.rerun()

        if submit_button:
            question = st.session_state.question_input.strip()
            if question:
                if question.lower() == "quiz":
                    st.subheader("📋 Revision Questions")
                    with st.spinner("Generating revision questions..."):
                        try:
                            questions = rag.generate_revision_questions(
                                st.session_state.selected_page["content"]
                            )
                            st.text_area(
                                "Generated Questions", value=questions, height=400
                            )
                            mark_page_revised(st.session_state.selected_page["title"])
                            st.session_state.answer_history.append(
                                {"type": "quiz", "content": questions}
                            )
                            st.session_state.question_input = ""
                        except Exception as e:
                            st.error(f"Error generating questions: {str(e)}")
                else:
                    st.subheader("💡 Answer")
                    answer_container = st.empty()
                    try:
                        full_answer = ""
                        with st.spinner("Thinking..."):
                            for chunk in rag.ask(question, stream=True):
                                full_answer += chunk
                                answer_container.markdown(full_answer)

                        st.session_state.answer_history.append(
                            {"type": "qa", "question": question, "answer": full_answer}
                        )
                        mark_page_revised(st.session_state.selected_page["title"])
                        st.session_state.question_input = ""
                    except Exception as e:
                        st.error(f"Error generating answer: {str(e)}")

        if st.session_state.answer_history:
            st.subheader("📜 History")
            for i, item in enumerate(reversed(st.session_state.answer_history)):
                with st.expander(
                    f"{'Quiz #' if item['type'] == 'quiz' else 'Q: ' + item['question']}",
                    expanded=(i == 0),
                ):
                    if item["type"] == "quiz":
                        st.text_area(
                            f"Quiz {len(st.session_state.answer_history) - i}",
                            value=item["content"],
                            height=200,
                        )
                    else:
                        st.markdown(item["answer"])
    else:
        st.info("👈 Select a page from your Notion workspace to get started.")
