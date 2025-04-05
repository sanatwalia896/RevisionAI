# revisionai_rag.py (LangChain v0.3+ with chat history and Qdrant)
import os
from langchain_community.vectorstores import Qdrant
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain_groq import ChatGroq
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance


class RevisionRAG:
    def __init__(
        self,
        groq_api_key: str,
        qdrant_url: str,
        qdrant_api_key: str,
        collection_name: str = "revisionai",
        embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    ):
        self.embedding = HuggingFaceEmbeddings(model_name=embedding_model_name)
        self.llm = ChatGroq(api_key=groq_api_key, model_name="llama3-8b-8192")
        self.qdrant_url = qdrant_url
        self.qdrant_api_key = qdrant_api_key
        self.collection_name = collection_name
        self.vectorstore = None
        self.retriever = None
        self.qa_with_history = None

        # ✅ Correctly use constructor params instead of os.getenv
        self.qdrant_client = QdrantClient(
            url=self.qdrant_url,
            api_key=self.qdrant_api_key,
        )

        self._ensure_qdrant_collection()

    def _ensure_qdrant_collection(self):
        dim = len(self.embedding.embed_query("test"))  # get embedding size dynamically
        existing = [
            col.name for col in self.qdrant_client.get_collections().collections
        ]
        if self.collection_name not in existing:
            self.qdrant_client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
            )

    def build_rag_from_pages(self, pages: list):
        for page in pages:
            self.refresh_page_in_vectorstore(page)

        self.vectorstore = Qdrant(
            client=self.qdrant_client,
            collection_name=self.collection_name,
            embeddings=self.embedding,
        )

        self.retriever = self.vectorstore.as_retriever()

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

    def refresh_page_in_vectorstore(self, page: dict):
        # Delete old vectors for this page
        self.qdrant_client.delete(
            collection_name=self.collection_name,
            filter={"must": [{"key": "page_title", "match": {"value": page["title"]}}]},
        )

        # Chunk updated content
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = splitter.split_text(page["content"])

        docs = [
            Document(page_content=chunk, metadata={"page_title": page["title"]})
            for chunk in chunks
        ]

        # Re-upload to Qdrant using constructor params
        Qdrant.from_documents(
            documents=docs,
            embedding=self.embedding,
            url=self.qdrant_url,
            api_key=self.qdrant_api_key,
            collection_name=self.collection_name,
        )

        print(f"🔁 Refreshed page in vectorstore: {page['title']}")

    def ask(self, question: str, session_id: str = "default") -> str:
        if self.qa_with_history is None:
            return "RAG pipeline not initialized."
        response = self.qa_with_history.invoke(
            {"query": question},
            config={"configurable": {"session_id": session_id}},
        )
        return response

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
            all_questions.append(response)

        return "\n\n".join(all_questions)
