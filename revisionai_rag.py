import os
import json
import datetime
from hashlib import md5
from uuid import uuid4
from pathlib import Path

from dotenv import load_dotenv
from langchain_qdrant import QdrantVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings

from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from langchain_groq import ChatGroq
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage
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


class RevisionRAG:
    def __init__(
        self,
        groq_api_key: str,
        qdrant_url: str,
        qdrant_api_key: str,
        collection_name: str = "revisionai",
    ):
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        load_dotenv()

        self.embedding = HuggingFaceInferenceAPIEmbeddings(
            api_key=os.getenv("HUGGINGFACE_TOKEN"),
            model_name="sentence-transformers/all-MiniLM-L6-v2",
        )

        self.llm = ChatGroq(api_key=groq_api_key, model_name="llama3-8b-8192")
        self.qdrant_url = qdrant_url
        self.qdrant_api_key = qdrant_api_key
        self.collection_name = collection_name
        self.vectorstore = None
        self.retriever = None
        self.qa_chain = None
        self.qa_with_history = None
        self.content_hashes = self._load_content_hashes()
        self.current_topic = "all"

        self.qdrant_client = QdrantClient(
            url=self.qdrant_url,
            api_key=self.qdrant_api_key,
        )

        self._ensure_qdrant_collection()
        self._initialize_vectorstore()

    def _load_content_hashes(self):
        if os.path.exists(HASH_CACHE_FILE):
            with open(HASH_CACHE_FILE, "r") as f:
                return json.load(f)
        return {}

    def _save_content_hashes(self):
        with open(HASH_CACHE_FILE, "w") as f:
            json.dump(self.content_hashes, f, indent=2)

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

    def build_rag_from_pages(self, pages: list):
        updated_pages = 0
        unchanged_pages = 0

        for page in pages:
            current_hash = self._compute_content_hash(page["content"])
            if (
                page["title"] in self.content_hashes
                and self.content_hashes[page["title"]] == current_hash
            ):
                print(f"🔁 No changes detected for page: {page['title']}")
                unchanged_pages += 1
                continue

            self.refresh_page_in_vectorstore(page)
            self.content_hashes[page["title"]] = current_hash
            updated_pages += 1

        self._save_content_hashes()

        print(f"✅ Updated {updated_pages} pages, {unchanged_pages} pages unchanged")

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

        return updated_pages > 0

    def refresh_page_in_vectorstore(self, page: dict):
        print(f"⚙️ Updating vectors for page: {page['title']}")

        self.qdrant_client.delete(
            collection_name=self.collection_name,
            points_selector=Filter(
                must=[
                    FieldCondition(
                        key="page_title", match=MatchValue(value=page["title"])
                    )
                ]
            ),
        )

        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = splitter.split_text(page["content"])

        docs = []
        for chunk in chunks:
            docs.append(
                Document(page_content=chunk, metadata={"page_title": page["title"]})
            )

        QdrantVectorStore.from_documents(
            documents=docs,
            embedding=self.embedding,
            url=self.qdrant_url,
            api_key=self.qdrant_api_key,
            collection_name=self.collection_name,
        )

        print(f"✅ Refreshed page in vectorstore: {page['title']} ({len(docs)} chunks)")

    def ask(self, question: str, session_id: str = "default") -> str:
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

        response = self.qa_with_history.invoke(
            {"query": question},
            config={"configurable": {"session_id": session_id}},
        )
        return response

    def ask_streaming(self, question: str, session_id: str = "default"):
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

        config: RunnableConfig = {
            "configurable": {"session_id": session_id},
            "stream": True,
        }
        return self.qa_with_history.stream({"query": question}, config=config)

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
            if isinstance(response, AIMessage):
                all_questions.append(response.content)
            else:
                all_questions.append(str(response))

        return "\n\n".join(all_questions)

    def extract_topic_from_title(self, title: str) -> str:
        if ":" in title:
            return title.split(":")[0].strip().lower()
        elif "-" in title:
            return title.split("-")[0].strip().lower()
        else:
            return "general"

    def get_available_topics(self) -> list:
        topics = set()
        for title in self.content_hashes.keys():
            topics.add(self.extract_topic_from_title(title))
        return sorted(list(topics))

    def set_topic(self, topic: str):
        self.current_topic = topic.lower() if topic else "all"

    def filter_pages_by_topic(self, all_pages: list, topic: str) -> list:
        if not topic or topic.lower() == "all":
            return all_pages
        return [
            page
            for page in all_pages
            if self.extract_topic_from_title(page["title"]) == topic.lower()
        ]


def check_due_revisions(display=True):
    with open("revision_schedule.json", "r") as f:
        schedule = json.load(f)

    due_pages = []
    today = datetime.date.today()

    for page_title, last_revised in schedule.items():
        last_date = datetime.datetime.strptime(last_revised, "%Y-%m-%d").date()
        days_since = (today - last_date).days
        if days_since >= 3:
            due_pages.append((page_title, days_since))

    if display:
        if due_pages:
            print("\n🔔 Pages due for revision:")
            for page, days in due_pages:
                print(f"• {page} (Last revised {days} days ago)")
        else:
            print("\n✅ No pages due for revision.")

    return due_pages
