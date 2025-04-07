import os
import json
import datetime
from hashlib import md5
from pathlib import Path
from dotenv import load_dotenv

from langchain_qdrant import QdrantVectorStore
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_groq import ChatGroq
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig

from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams,
    Distance,
    Filter,
    FieldCondition,
    MatchValue,
)

load_dotenv()

HASH_CACHE_FILE = "page_content_hashes.json"
REVISION_SCHEDULE_FILE = "revision_schedule.json"


class RevisionRAG:
    def __init__(
        self,
        groq_api_key: str,
        qdrant_url: str,
        qdrant_api_key: str,
        collection_name: str = "revisionai",
    ):
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        self.embedding = HuggingFaceInferenceAPIEmbeddings(
            api_key=os.getenv("HUGGINGFACE_TOKEN"),
            model_name="sentence-transformers/all-MiniLM-L6-v2",
        )

        self.llm = ChatGroq(api_key=groq_api_key, model_name="llama3-8b-8192")
        self.qdrant_client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
        self.collection_name = collection_name

        self.content_hashes = self._load_json(HASH_CACHE_FILE)
        self.current_topic = "all"
        self.qa_with_history = None

        self._ensure_qdrant_collection()
        self._initialize_vectorstore()

    def _load_json(self, path):
        if os.path.exists(path):
            with open(path, "r") as f:
                return json.load(f)
        return {}

    def _save_json(self, path, data):
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def _compute_content_hash(self, content):
        return md5(content.encode("utf-8")).hexdigest()

    def _ensure_qdrant_collection(self):
        collections = self.qdrant_client.get_collections().collections
        if self.collection_name not in [col.name for col in collections]:
            self.qdrant_client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=384, distance=Distance.COSINE),
            )

    def _initialize_vectorstore(self):
        self.vectorstore = QdrantVectorStore(
            client=self.qdrant_client,
            collection_name=self.collection_name,
            embedding=self.embedding,
        )
        self.retriever = self.vectorstore.as_retriever()

    def _ensure_qa_with_history(self):
        if self.qa_with_history is None:
            base_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                retriever=self.retriever,
                chain_type="stuff",
            )
            self.qa_with_history = RunnableWithMessageHistory(
                base_chain,
                lambda session_id: ChatMessageHistory(),
                input_messages_key="query",
                history_messages_key="history",
            )

    def build_rag_from_pages(self, pages: list):
        updated, unchanged = 0, 0
        for page in pages:
            title, content = page["title"], page["content"]
            content_hash = self._compute_content_hash(content)

            if self.content_hashes.get(title) == content_hash:
                print(f"🔁 No changes detected for page: {title}")
                unchanged += 1
                continue

            self._refresh_page(title, content)
            self.content_hashes[title] = content_hash
            updated += 1

        self._save_json(HASH_CACHE_FILE, self.content_hashes)
        print(f"✅ Updated {updated} pages, {unchanged} pages unchanged")
        self._ensure_qa_with_history()
        return updated > 0

    def _refresh_page(self, title, content):
        print(f"⚙️ Updating vectors for page: {title}")

        self.qdrant_client.delete(
            collection_name=self.collection_name,
            points_selector=Filter(
                must=[FieldCondition(key="page_title", match=MatchValue(value=title))]
            ),
        )

        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = splitter.split_text(content)

        docs = [
            Document(page_content=chunk, metadata={"page_title": title})
            for chunk in chunks
        ]

        QdrantVectorStore.from_documents(
            documents=docs,
            embedding=self.embedding,
            url=self.qdrant_client.url,
            api_key=self.qdrant_client.api_key,
            collection_name=self.collection_name,
        )

        print(f"✅ Refreshed page in vectorstore: {title} ({len(docs)} chunks)")

    def ask(self, question: str, session_id: str = "default", stream: bool = False):
        self._ensure_qa_with_history()
        config: RunnableConfig = {
            "configurable": {"session_id": session_id},
        }

        if stream:
            return self.qa_with_history.stream({"query": question}, config=config)
        else:
            return self.qa_with_history.invoke({"query": question}, config=config)

    def generate_revision_questions(self, content: str) -> str:
        splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=100)
        chunks = splitter.split_text(content)
        all_questions = []

        for chunk in chunks:
            prompt = (
                "Generate the following types of revision questions based on the content below:\n"
                "1. 3 Multiple Choice Questions\n"
                "2. 3 One Word Answer Questions\n"
                "3. 2 Short Answer Questions\n"
                "4. 1 Long Answer Question\n"
                "5. If the content is code-related, generate a 'Explain the Code' question\n"
                f"\nContent:\n{chunk}\n"
            )
            response = self.llm.invoke(prompt)
            all_questions.append(
                response.content if isinstance(response, AIMessage) else str(response)
            )

        return "\n\n".join(all_questions)

    def extract_topic_from_title(self, title: str) -> str:
        if ":" in title:
            return title.split(":")[0].strip().lower()
        elif "-" in title:
            return title.split("-")[0].strip().lower()
        return "general"

    def get_available_topics(self) -> list:
        return sorted({self.extract_topic_from_title(t) for t in self.content_hashes})

    def set_topic(self, topic: str):
        self.current_topic = topic.lower() if topic else "all"

    def filter_pages_by_topic(self, pages: list, topic: str) -> list:
        if not topic or topic.lower() == "all":
            return pages
        return [
            p
            for p in pages
            if self.extract_topic_from_title(p["title"]) == topic.lower()
        ]


def check_due_revisions(display=True):
    schedule = {}
    if os.path.exists(REVISION_SCHEDULE_FILE):
        with open(REVISION_SCHEDULE_FILE, "r") as f:
            schedule = json.load(f)

    today = datetime.date.today()
    due_pages = []

    for title, last in schedule.items():
        last_date = datetime.datetime.strptime(last, "%Y-%m-%d").date()
        days_since = (today - last_date).days
        if days_since >= 3:
            due_pages.append((title, days_since))

    if display:
        if due_pages:
            print("\n🔔 Pages due for revision:")
            for title, days in due_pages:
                print(f"• {title} (Last revised {days} days ago)")
        else:
            print("\n✅ No pages due for revision.")

    return due_pages
