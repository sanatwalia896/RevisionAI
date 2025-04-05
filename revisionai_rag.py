# revisionai_rag.py

from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain_groq import ChatGroq


class RevisionRAG:
    def __init__(
        self,
        groq_api_key: str,
        embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    ):
        self.embedding = HuggingFaceEmbeddings(model_name=embedding_model_name)
        self.llm = ChatGroq(api_key=groq_api_key, model_name="mixtral-8x7b-32768")
        self.vectorstore = None
        self.retriever = None
        self.qa_chain = None

    def build_rag_from_pages(self, pages: list):
        all_text = "\n\n".join([p["content"] for p in pages])
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        docs = [Document(page_content=chunk) for chunk in splitter.split_text(all_text)]
        self.vectorstore = FAISS.from_documents(docs, self.embedding)
        self.retriever = self.vectorstore.as_retriever()
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm, retriever=self.retriever, chain_type="stuff"
        )

    def ask(self, question: str) -> str:
        if self.qa_chain is None:
            return "RAG pipeline not initialized."
        return self.qa_chain.run(question)

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
