# revisionai_rag.py (LangChain v0.3+ with chat history)
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain_groq import ChatGroq

from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage


class RevisionRAG:
    def __init__(
        self,
        groq_api_key: str,
        embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    ):
        self.embedding = HuggingFaceEmbeddings(model_name=embedding_model_name)
        self.llm = ChatGroq(api_key=groq_api_key, model_name="llama3-8b-8192")
        self.vectorstore = None
        self.retriever = None
        self.qa_chain = None
        self.qa_with_history = None

    def build_rag_from_pages(self, pages: list):
        all_text = "\n\n".join([p["content"] for p in pages])
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        docs = [Document(page_content=chunk) for chunk in splitter.split_text(all_text)]
        self.vectorstore = FAISS.from_documents(docs, self.embedding)
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

    def ask(self, question: str, session_id: str = "default") -> str:
        if self.qa_with_history is None:
            return "RAG pipeline not initialized."
        response = self.qa_with_history.invoke(
            {"query": question},
            config={"configurable": {"session_id": session_id}},
        )
        return response

    def generate_revision_questions(self, content: str) -> dict:
        prompt = (
            "Generate the following types of revision questions based on the content below:\n"
            "1. 3 Multiple Choice Questions\n"
            "2. 3 One Word Answer Questions\n"
            "3. 2 Short Answer Questions\n"
            "4. 1 Long Answer Question\n"
            "5. If the content is code-related, generate a 'Explain the Code' question\n"
            f"\nContent:\n{content}\n"
        )
        return self.llm.invoke(prompt)
