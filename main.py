# main.py

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

# Show available pages
print("\nAvailable Pages:")
for i, page in enumerate(all_pages):
    print(f"{i + 1}. {page['title']}")

selection = int(input("\nSelect page number: ")) - 1
selected = all_pages[selection]

print(
    f"\n🧠 Building RAG from: {selected['title']} ({len(selected['content'].split())} words)\n"
)

# Initialize and build RAG pipeline once for the selected page
rag = RevisionRAG(
    groq_api_key=groq_api_key,
    qdrant_url=qdrant_url,
    qdrant_api_key=qdrant_api_key,
)

rag.build_rag_from_pages([selected])

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
