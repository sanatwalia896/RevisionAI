import os
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

# Initialize the RAG system first, so it loads cached content hashes
rag = RevisionRAG(
    groq_api_key=groq_api_key,
    qdrant_url=qdrant_url,
    qdrant_api_key=qdrant_api_key,
)

# Load Notion pages from cache or refresh
reader = NotionPageLoader(notion_token)

# Ask to sync and update the cache
sync = input("Do you want to sync Notion pages? (y/n): ").lower()
if sync == "y":
    reader.refresh_and_cache_pages()

all_pages = reader.get_all_page_contents()

if not all_pages:
    print("No pages found in cache. Please sync first.")
    exit()

# Show revision reminders
check_due_revisions()

# Option to update all vectorstore entries
update_all = input("\nDo you want to check all pages for updates? (y/n): ").lower()
if update_all == "y":
    print("\n🔄 Checking all pages for updates and building vectors...")
    rag.build_rag_from_pages(all_pages)

# Get available topics
topics = rag.get_available_topics()
if topics and len(topics) > 1:  # Only show topic selection if multiple topics exist
    print("\nAvailable Topics:")
    print("0. All Topics")
    for i, topic in enumerate(topics):
        print(f"{i + 1}. {topic}")

    topic_selection = input("\nSelect topic number (or press Enter for all): ")
    if topic_selection.strip():
        selected_topic = (
            "all" if topic_selection == "0" else topics[int(topic_selection) - 1]
        )
        rag.set_topic(selected_topic)
        print(f"🔍 Filtering for topic: {selected_topic}")
    else:
        rag.set_topic("all")

# Show available pages (filtered by topic if selected)
current_topic = rag.current_topic
print("\nAvailable Pages:")
filtered_pages = []
for i, page in enumerate(all_pages):
    page_topic = rag.extract_topic_from_title(page["title"])
    if current_topic == "all" or current_topic is None or page_topic == current_topic:
        filtered_pages.append(page)
        print(f"{len(filtered_pages)}. {page['title']} (Topic: {page_topic})")

if not filtered_pages:
    print(f"No pages found for topic: {current_topic}")
    print("Showing all pages instead:")
    filtered_pages = all_pages
    for i, page in enumerate(filtered_pages):
        page_topic = rag.extract_topic_from_title(page["title"])
        print(f"{i + 1}. {page['title']} (Topic: {page_topic})")
    rag.set_topic("all")

selection = int(input("\nSelect page number: ")) - 1
selected = filtered_pages[selection]
selected_topic = rag.extract_topic_from_title(selected["title"])

print(
    f"\n🧠 Selected: {selected['title']} (Topic: {selected_topic}, {len(selected['content'].split())} words)\n"
)

# Only build RAG for the selected page if we didn't update all pages already
if update_all != "y":
    rag.build_rag_from_pages([selected], selected_topic)

# Make sure we're using the right topic for the selected page
rag.set_topic(selected_topic)

while True:
    q = input("Ask a question (or 'quiz' for revision, 'exit' to quit): ")
    if q.lower() in {"exit", "quit"}:
        break
    elif q.lower() == "quiz":
        questions = rag.generate_revision_questions(selected["content"])
        print("\n📋 Revision Questions:\n")
        print(questions)
        mark_page_revised(selected["title"])  # Mark as revised after quiz
    else:
        answer = rag.ask(q)
        print("\n💡 Answer:\n", answer)
        mark_page_revised(selected["title"])  # Mark as revised after asking
