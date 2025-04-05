# main.py

import os
from dotenv import load_dotenv
from revisionai_notion import NotionPageLoader
from revisionai_rag import RevisionRAG
from revision_scheduler import check_due_revisions, mark_page_revised

load_dotenv()

notion_token = os.getenv("NOTION_TOKEN")
groq_api_key = os.getenv("GROQ_API_KEY")

# Load content from Notion
reader = NotionPageLoader(notion_token)
all_pages = reader.get_all_page_contents()

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
rag = RevisionRAG(groq_api_key)
rag.build_rag_from_pages([selected])

while True:
    q = input("Ask a question (or 'quiz' for revision, 'exit' to quit): ")
    if q.lower() in {"exit", "quit"}:
        break
    elif q.lower() == "quiz":
        questions = rag.generate_revision_questions(selected["content"])
        print("📋 Revision Questions:\n", questions)
        mark_page_revised(selected["title"])  # Mark as revised after quiz
    else:
        answer = rag.ask(q)
        print("💡", answer)
        mark_page_revised(selected["title"])  # Mark as revised after asking
